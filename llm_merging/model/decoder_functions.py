import torch
import torch.nn.functional as F

def predict_multiple_choice(
    transformer, input_tokenizer, target_tokenizer, batch
): 
    tokenized_batch = tokenize_batch(input_tokenizer, target_tokenizer, batch, transformer.device)
    output = transformer(
        input_ids=tokenized_batch["input_ids"],
        attention_mask=tokenized_batch["input_mask"],
        use_cache=True,
    )
    past_key_values = output.past_key_values

    num_answer_choices = (
        tokenized_batch["answer_choices_ids"].shape[0]
        // tokenized_batch["input_mask"].shape[0]
    )

    '''
    Expand the input_mask and past_key_values since these are the same and can be repeated for the different answer choices within an example 
    '''

    batch_size, max_input_len = tokenized_batch["input_mask"].shape
    expanded_input_mask = torch.repeat_interleave(tokenized_batch["input_mask"], num_answer_choices, dim=0)

    expanded_past_key_valyes = []
    for pastKeyValues_perLayer in past_key_values:
        list_broadcast_pastKeyValues_perLayer = []
        for key_or_value in pastKeyValues_perLayer:
            # This is for keys or values which have dimension [batch_size, max_input_len, num_heads, head_dim]
            # This is the standard for Hugging Face.
            if len(key_or_value.shape) == 4:
                list_broadcast_pastKeyValues_perLayer.append(
                    torch.repeat_interleave(key_or_value, num_answer_choices, dim=0)
                )
            # This is for keys or values which have dimension [batch_size x num_heads, head_dim, max_input_len].
            # This is what is used for BLOOM in transformers == 4.22.0
            elif len(key_or_value.shape) == 3:
                num_heads = key_or_value.shape[0] // batch_size
                flatten_keyOrValue = key_or_value.reshape(
                    ((batch_size, num_heads) + key_or_value.shape[1:])
                )
                broadcast_flatten_keyOrValue = torch.repeat_interleave(
                    flatten_keyOrValue, num_answer_choices, dim=0
                )
                list_broadcast_pastKeyValues_perLayer.append(
                    broadcast_flatten_keyOrValue.flatten(0, 1)
                )
            else:
                raise ValueError(
                    f"Invalid cached key or value shape: ", key_or_value.shape
                )

        expanded_past_key_valyes.append(
            tuple(list_broadcast_pastKeyValues_perLayer)
        )


    # Combine the input mask and choice mask so the model knows which cached input representations
    # are padded when conditioning on the cached input representations.
    # [batch_size x num_choices, max_input_len + max_choice_len]
    combined_mask = torch.cat(
        [expanded_input_mask, tokenized_batch["answer_choices_mask"]], dim=1
    )

    # WARNING: The loss at transformer_outputs[0] is not valid, since allChoices_ids uses a
    # pad token of 0 and so the loss will not be ignored for the pad tokens
    transformer_outputs = transformer(
        input_ids=tokenized_batch["answer_choices_ids"],
        attention_mask=combined_mask,
        past_key_values=expanded_past_key_valyes,
        use_cache=True,
    )

    # We used the logits for all choices to compute the log probs per example since
    # the loss returned in transformer_outputs will average the negative log probs across examples
    # [batch_size x num_choices, max_choice_len, vocab_size]
    answer_choice_ids_logits = transformer_outputs.logits.float()
    vocab_size = answer_choice_ids_logits.shape[-1]

    # Shift the ids, masks, and logits to handle predicting the next token for the decoder. Note that we need to pass in the input_ids and cannot rely on HuggingFace automatically constructing the ids from the labels, since we need to pass in an attention mask to handle the cached input representations.
    shifted_answer_choice_ids_logits = answer_choice_ids_logits[..., :-1, :].contiguous()
    shifted_answer_choice_ids = tokenized_batch["answer_choices_ids"][
        ..., 1:
    ].contiguous()
    shifted_answer_choice_masks = tokenized_batch["answer_choices_mask"][
        ..., 1:
    ].contiguous()

    shifted_answer_choices_max_len = shifted_answer_choice_ids_logits.shape[1]
    vocab_size = shifted_answer_choice_ids_logits.shape[-1]

    # Compute the log probability of the ids for all choices with respect to the logits [batch_size x num_choices x (max_choice_len-1)]
    shifted_answer_choice_ids_log_probs = -F.cross_entropy(
        shifted_answer_choice_ids_logits.view(-1, vocab_size),
        shifted_answer_choice_ids.view(-1),
        reduction="none",
    )


    # [batch_size, num_answer_choices, answer_choices_max_len]
    shifted_answer_choice_ids_log_probs = shifted_answer_choice_ids_log_probs.reshape(
        -1, num_answer_choices, shifted_answer_choices_max_len
    )

    shifted_answer_choices_mask = shifted_answer_choice_masks.reshape(
        -1, num_answer_choices, shifted_answer_choices_max_len
    )

    answer_choice_log_probs = torch.sum(shifted_answer_choice_ids_log_probs * shifted_answer_choices_mask, dim=2)

    _, predicted_choice = torch.max(answer_choice_log_probs, dim=1)

    return predicted_choice, answer_choice_log_probs



def generate(
    transformer, 
    input_tokenizer,
    target_tokenizer,
    batch,
    max_gen_len
):
    tokenized_batch = tokenize_batch(input_tokenizer, target_tokenizer, batch, transformer.device)

    generation_output = transformer.generate(
        input_ids=tokenized_batch["input_ids"],
        attention_mask=tokenized_batch["input_mask"],
        max_new_tokens=max_gen_len,
        eos_token_id=input_tokenizer.eos_token_id,
        pad_token_id=input_tokenizer.pad_token_id,
        bos_token_id=input_tokenizer.bos_token_id,
        do_sample=False,
        return_dict_in_generate=True,
    )

    # Remove the original input ids from the generated ids to get just the generated ids 
    input_len = tokenized_batch[f"input_ids"].shape[-1]

    generated_ids = generation_output["sequences"][:, input_len:]

    generated_txt = input_tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )

    return generation_output["sequences"].cpu().numpy().tolist(), generated_txt

def tokenize_batch(input_tokenizer, target_tokenizer, batch, device):

    tokenized_batch = {}

    keys_to_tokenize_with_tokenizer = [("input", input_tokenizer), ("answer_choices", target_tokenizer), ("target", target_tokenizer)]


    # Tokenize keys which should be tokenized
    for key, tokenizer in keys_to_tokenize_with_tokenizer:
        if key in batch:
            # The values of the batch are normally a list of text.The exception is that for answer_choices, the values  is a list of list. We flatten this to a single list to pass is into the tokenizer 
            if key == "answer_choices":
                text = batch[key]
                for i, t in enumerate(text):
                    if t[0] == "[":
                        import ast
                        text[i] = ast.literal_eval(t)
                text = [item for list in text for item in list]
            else:
                text = batch[key]
                for i, t in enumerate(text):
                    if isinstance(t, (int, float)):
                        text[i] = str(t)
        tokenized_dict = tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation="longest_first",
        )

        input_ids = tokenized_dict["input_ids"]
        attention_mask = tokenized_dict["attention_mask"]

        if device is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

        tokenized_batch[f"{key}_ids"] = input_ids
        tokenized_batch[f"{key}_mask"] = attention_mask
    return tokenized_batch