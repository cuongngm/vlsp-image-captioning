import os, time
import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import argparse
import imageio
import cv2
from PIL import Image, ImageFile
from util.label_convert import LabelConvert
ImageFile.LOAD_TRUNCATED_IMAGES = True


def caption_image_beam_search(args, encoder, decoder, image_path, word_map):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """
    k = args.beam_size
    Caption_End = False
    vocab_size = len(word_map)
    # vocab_size = 64000

    # Read image and process
    try:
        img = imageio.imread(image_path)
    except:
        img = Image.open(image_path)
        img = np.array(img)
    # img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    if img.shape[2] == 4:
        print('bgra', image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = np.array(Image.fromarray(img).resize((256, 256)))
    # img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)
    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(-1)
    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # [1, num_pixels=196, encoder_dim]
    num_pixels = encoder_out.size(1)
    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
    
    # Tensor to store top k previous words at each step; now they're just <start>
    if args.decoder_mode == "lstm":
        # k_prev_words = torch.LongTensor([[word_map['<SOS>']]] * k).to(device)  # (k, 1)
        k_prev_words = torch.LongTensor([[0]] * k).to(device)  # (k, 1)
    elif args.decoder_mode == "transformer":
        k_prev_words = torch.LongTensor([[word_map['<SOS>']] * 52] * k).to(device)  # (k, 52)

    # Tensor to store top k sequences; now they're just <start>
    # seqs = torch.LongTensor([[word_map['<SOS>']]] * k).to(device)  # (k, 1)
    seqs = torch.LongTensor([[word_map['<SOS>']]] * k).to(device)  # (k, 1)
    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)
    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    if args.decoder_mode == "lstm":
        h, c = decoder.init_hidden_state(encoder_out)
    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        if args.decoder_mode == "lstm":
            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
            alpha = alpha.view(-1, enc_image_size, enc_image_size).unsqueeze(1)  # (s, 1, enc_image_size, enc_image_size)
            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe
            h, c = decoder.lstm(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
            scores = decoder.fc(h)  # (s, vocab_size)
        elif args.decoder_mode == "transformer":
            cap_len = torch.LongTensor([52]).repeat(k, 1)  # [s, 1]
            scores, _, _, alpha_dict, _ = decoder(encoder_out, k_prev_words, cap_len)
            scores = scores[:, step - 1, :].squeeze(1)  # [s, 1, vocab_size] -> [s, vocab_size]
            # choose the last layer, transformer decoder is comosed of a stack of 6 identical layers.
            alpha = alpha_dict["dec_enc_attns"][-1]  # [s, n_heads=8, len_q=52, len_k=196]
            # TODO: AVG Attention to Visualize
            # for i in range(len(alpha_dict["dec_enc_attns"])):
            #     n_heads = alpha_dict["dec_enc_attns"][i].size(1)
            #     for j in range(n_heads):
            #         pass
            # the second dim corresponds to the Multi-head attention = 8, now 0
            # the third dim corresponds to cur caption position
            alpha = alpha[:, 0, step-1, :].view(k, 1, enc_image_size, enc_image_size)  # [s, 1, enc_image_size, enc_image_size]
         
        scores = F.log_softmax(scores, dim=1)
        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)
        
        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)
        
        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds]], dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<EOS>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
    
        # Set aside complete sequences
        if len(complete_inds) > 0:
            Caption_End = True
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        if args.decoder_mode == "lstm":
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
        elif args.decoder_mode == "transformer":
            k_prev_words = k_prev_words[incomplete_inds]
            k_prev_words[:, :step + 1] = seqs  # [s, 52]
            # k_prev_words[:, step] = next_word_inds[incomplete_inds]  # [s, 52]

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1
    assert Caption_End
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image_Captioning')
    parser.add_argument('--img', '-i', default="./dataset/val2014/COCO_val2014_000000581886.jpg", help='path to image, file or folder')
    parser.add_argument('--checkpoint', '-m', default="./BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar", help='path to model')
    parser.add_argument('--word_map', '-wm', default="./dataset/generated_data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json",
                        help='path to word map JSON')
    parser.add_argument('--decoder_mode', default="lstm", help='which model does decoder use?')  # lstm or transformer
    parser.add_argument('--save_img_dir', '-p', default="./caption", help='path to save annotated img.')
    parser.add_argument('--beam_size', '-b', type=int, default=3, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()
    # print(encoder)
    # print(decoder)

    convert = LabelConvert(vocab_file='util/vocab_coco.txt')
    word_map = convert.vocab_mapper
    rev_word_map = convert.vocab_inverse_mapper
    print('load word map success!!')

    with open('dataset/vlsp_test/sample_submission.json', 'r') as f:
        datas = json.load(f)
        all_result = []
        for idx, data in enumerate(datas):
            each_result = dict()
            imgname = data['id']
            imgpath = os.path.join('dataset/vlsp_test/images_public_test', imgname)
            print(imgpath)
            with torch.no_grad():
                seq, alphas = caption_image_beam_search(args, encoder, decoder, imgpath, word_map)
            print(seq)
            
            words = [rev_word_map[ind] for ind in seq]
            print(words)
            words = words[1:-1]
            result = []
            for word in words:
                if word == '<UNK>':
                    continue
                if '_' in word:
                    word = word.split('_')
                else:
                    word = [word]
                result += word
            result = ' '.join(result)
            print(result)
            each_result['id'] = imgname
            each_result['captions'] = result
            all_result.append(each_result)

    with open('dataset/vlsp_test/results.json', 'w', encoding='utf-8') as fp:
        json.dump(all_result, fp, ensure_ascii=False)
