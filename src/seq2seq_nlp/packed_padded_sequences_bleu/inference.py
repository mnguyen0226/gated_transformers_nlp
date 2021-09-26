"""
Our translate_sentence will do the following:

    ensure our model is in evaluation mode, which it should always be for inference
    tokenize the source sentence if it has not been tokenized (is a string)
    numericalize the source sentence
    convert it to a tensor and add a batch dimension
    get the length of the source sentence and convert to a tensor
    feed the source sentence into the encoder
    create the mask for the source sentence
    create a list to hold the output sentence, initialized with an <sos> token
    create a tensor to hold the attention values
    while we have not hit a maximum length
        get the input tensor, which should be either <sos> or the last predicted token
        feed the input, all encoder outputs, hidden state and mask into the decoder
        store attention values
        get the predicted next token
        add prediction to current output sentence prediction
        break if the prediction was an <eos> token
    convert the output sentence from indexes to tokens
    return the output sentence (with the <sos> token removed) and the attention values over the sequence
"""
from main import *
from test_model import *

model.load_state_dict(torch.load("tut4-model.pt", map_location=torch.device("cpu")))


def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):

    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load("de")
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    src_len = torch.LongTensor([len(src_indexes)])

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    mask = model.create_mask(src_tensor)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)

    for i in range(max_len):

        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output, hidden, attention = model.decoder(
                trg_tensor, hidden, encoder_outputs, mask
            )

        attentions[i] = attention

        pred_token = output.argmax(1).item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attentions[: len(trg_tokens) - 1]


def display_attention(sentence, translation, attention):
    """ Next, we'll make a function that displays the model's attention over the source sentence for each target token generated. """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    attention = attention.squeeze(1).cpu().detach().numpy()

    cax = ax.matshow(attention, cmap="bone")

    ax.tick_params(labelsize=15)

    x_ticks = [""] + ["<sos>"] + [t.lower() for t in sentence] + ["<eos>"]
    y_ticks = [""] + translation

    ax.set_xticklabels(x_ticks, rotation=45)
    ax.set_yticklabels(y_ticks)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


def inference_training_set():
    """ he lighter the square at the intersection between two words, the more attention the model gave to that source word when translating that target word. """
    example_idx = 12

    src = vars(train_data.examples[example_idx])["src"]
    trg = vars(train_data.examples[example_idx])["trg"]

    print(f"src = {src}")
    print(f"trg = {trg}")

    translation, attention = translate_sentence(src, SRC, TRG, model, device)

    print(f"predicted trg = {translation}")

    display_attention(src, translation, attention)


def inference_testing_set():
    example_idx = 18

    src = vars(test_data.examples[example_idx])["src"]
    trg = vars(test_data.examples[example_idx])["trg"]

    print(f"src = {src}")
    print(f"trg = {trg}")

    translation, attention = translate_sentence(src, SRC, TRG, model, device)

    print(f"predicted trg = {translation}")

    display_attention(src, translation, attention)


if __name__ == "__main__":
    inference_training_set()
    inference_testing_set()
