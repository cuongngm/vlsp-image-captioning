import imageio
from PIL import Image
import numpy as np
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# transformer.device = torch.device("cpu")
# models.device = torch.device("cpu")
print(device)

# Load model
checkpoint = torch.load(args.checkpoint, map_location=str(device))
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

vocab_size = 64000

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
imgs = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)

features = decoder.bert(k_prev_words)
embeddings = features['last_hidden_state'].squeeze(1)

awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
alpha = alpha.view(-1, enc_image_size, enc_image_size).unsqueeze(1)
gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
awe = gate * awe
h, c = decoder.lstm(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
scores = decoder.fc(h)  # (s, vocab_size)

