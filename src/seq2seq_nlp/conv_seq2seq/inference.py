""" 

"""
from main import *
import torchtext


def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):

    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load("de_core_news_sm")
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_conved, encoder_combined = model.encoder(src_tensor)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        with torch.no_grad():
            output, attention = model.decoder(
                trg_tensor, encoder_conved, encoder_combined
            )

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention


def display_attention(sentence, translation, attention):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    attention = attention.squeeze(0).cpu().detach().numpy()

    cax = ax.matshow(attention, cmap="bone")

    ax.tick_params(labelsize=15)
    ax.set_xticklabels(
        [""] + ["<sos>"] + [t.lower() for t in sentence] + ["<eos>"], rotation=45
    )
    ax.set_yticklabels([""] + translation)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


def test_training_data():

    example_idx = 2

    src = vars(train_data.examples[example_idx])["src"]
    trg = vars(train_data.examples[example_idx])["trg"]

    print(f"src = {src}")
    print(f"trg = {trg}")

    translation, attention = translate_sentence(src, SRC, TRG, model, device)

    print(f"predicted trg = {translation}")


def test_validating_data():
    example_idx = 2

    src = vars(valid_data.examples[example_idx])["src"]
    trg = vars(valid_data.examples[example_idx])["trg"]

    print(f"src = {src}")
    print(f"trg = {trg}")

    translation, attention = translate_sentence(src, SRC, TRG, model, device)

    print(f"predicted trg = {translation}")

    display_attention(src, translation, attention)


def test_testing_data():
    example_idx = 4

    src = vars(test_data.examples[example_idx])["src"]
    trg = vars(test_data.examples[example_idx])["trg"]

    print(f"src = {src}")
    print(f"trg = {trg}")

    translation, attention = translate_sentence(src, SRC, TRG, model, device)

    print(f"predicted trg = {translation}")

    display_attention(src, translation, attention)


from torchtext.data.metrics import bleu_score


def calculate_bleu(data, src_field, trg_field, model, device, max_len=50):

    trgs = []
    pred_trgs = []

    for datum in data:

        src = vars(datum)["src"]
        trg = vars(datum)["trg"]

        pred_trg, _ = translate_sentence(
            src, src_field, trg_field, model, device, max_len
        )

        # cut off <eos> token
        pred_trg = pred_trg[:-1]

        pred_trgs.append(pred_trg)
        trgs.append([trg])

    return bleu_score(pred_trgs, trgs)


def bleu():
    bleu_score = calculate_bleu(test_data, SRC, TRG, model, device)

    print(f"BLEU score = {bleu_score*100:.2f}")


if __name__ == "__main__":
    test_testing_data()
    test_training_data()
    test_validating_data()
    bleu()
