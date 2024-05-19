import torch
import torch.nn.functional as F


def compute_loss(transformer, tokenizer, batch):

    tokenized_batch = tokenize_batch(tokenizer, batch, transformer.device)

    transformer_outputs = transformer(
        input_ids=tokenized_batch["input_ids"],
        attention_mask=tokenized_batch["input_mask"],
        labels=tokenized_batch["target_ids"],
    )

    # [batch_size, max_target_len, vocab_size]
    target_logits = transformer_outputs[1].float()
    vocab_size = target_logits.shape[-1]

    # Compute the log probability of the ids for all choices with respect to the logits
    # [batch_size x max_target_len]
    negative_log_probs = F.cross_entropy(
        target_logits.reshape(-1, vocab_size),
        tokenized_batch["target_ids"].reshape(-1),
        reduction="none",
    )

    # Zero out log_probs for target_ids with no loss
    target_mask = tokenized_batch["target_mask"].reshape(-1)
    
    
    sum_negative_log_prob = torch.sum(
        negative_log_probs * target_mask
    )

    loss = sum_negative_log_prob / torch.sum(
            target_mask
        )

    return loss

def predict_multiple_choice(
    transformer, tokenizer, batch
):
    tokenized_batch = tokenize_batch(tokenizer, batch, transformer.device)


    encoder_outputs = transformer.get_encoder()(
        tokenized_batch["input_ids"],
        tokenized_batch["input_mask"],
    )

    # The answer_choices is the flattened batch of answer choices. To get the number of answer choices per example, we divide the total number of answer choices in a batch by the batch size. 
    num_answer_choices = (
        tokenized_batch["answer_choices_ids"].shape[0] // tokenized_batch["input_mask"].shape[0]
    )

    '''Expand the input_mask and encoder_outputs since these are the same and can be repeated for the different answer choices within an example 
    '''
    # [batch_size x num_choices, max_input_len]
    expanded_input_mask = torch.repeat_interleave(tokenized_batch["input_mask"], num_answer_choices, dim=0)
    # BaseModelOutput object from HuggingFace where the first element is the hidden states of the encoder at the last layer 
    # [batch_size x num_choices, max_input_len, ff_dim]
    expanded_encoder_outputs = (
        torch.repeat_interleave(encoder_outputs[0], num_answer_choices, dim=0),
    )


    # WARNING: The loss at transformer_outputs[0] is not valid, since answer_choices_ids uses a pad token of 0 (while loss expects a pad token of -100) so the loss will not be ignored for the pad tokens. 
    # The input mask is passed in for the cross encoder-decoder attention.
    transformer_outputs = transformer(
        attention_mask=expanded_input_mask,
        encoder_outputs=expanded_encoder_outputs,
        labels=tokenized_batch["answer_choices_ids"],
    )

    # We used the logits for all choices to compute the log probs per example since the loss returned in transformer_outputs will average the negative log probs across examples
    # [batch_size x num_choices, max_choice_len, vocab_size]
    answer_choice_ids_logits = transformer_outputs[1].float()
    answer_choices_max_len = answer_choice_ids_logits.shape[1]
    vocab_size = answer_choice_ids_logits.shape[-1]

    # Compute the log probability of the ids for all choices with respect to the logits
    # [batch_size x num_choices x max_choice_len]
    answer_choices_ids_log_probs = -F.cross_entropy(
        answer_choice_ids_logits.view(-1, vocab_size),
        tokenized_batch["answer_choices_ids"].view(-1),
        reduction="none",
    )

    # [batch_size, num_answer_choices, answer_choices_max_len]
    answer_choices_ids_log_probs = answer_choices_ids_log_probs.reshape(
        -1, num_answer_choices, answer_choices_max_len
    )

    answer_choices_mask = tokenized_batch["answer_choices_mask"].reshape(
        -1, num_answer_choices, answer_choices_max_len
    )

    answer_choice_log_probs = torch.sum(answer_choices_ids_log_probs * answer_choices_mask, dim=2)

    _, predicted_choice = torch.max(answer_choice_log_probs, dim=1)

    return predicted_choice, answer_choice_log_probs


def generate(
    transformer, 
    tokenizer,
    batch,
    max_gen_len
):
    tokenized_batch = tokenize_batch(tokenizer, batch, transformer.device)

    generation_output = transformer.generate(
        input_ids=tokenized_batch["input_ids"],
        attention_mask=tokenized_batch["input_mask"],
        max_new_tokens=max_gen_len,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        do_sample=False,
        return_dict_in_generate=True,
    )
    generated_txt = tokenizer.batch_decode(
        generation_output["sequences"], skip_special_tokens=True
    )

    return generation_output["sequences"].cpu().numpy().tolist(), generated_txt

def tokenize_batch(tokenizer, batch, device):        

    tokenized_batch = {}

    # encoder decoder models pad to the right 
    tokenizer.padding_side = "right"

    keys_to_tokenize = ["input", "answer_choices", "target"]

    for key in keys_to_tokenize:
        if key in batch:
            # The values of the batch are normally a list of text.The exception is that for answer_choices, the values  is a list of list. We flatten this to a single list to pass is into the tokenizer 
            if key == "answer_choices":
                text = [item for list in batch[key] for item in list]
            else:
                text = batch[key]

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
