import io
from PIL import Image
import boto3

client = boto3.client('rekognition')

def get_correct_label(labels):
    for label in labels:
        name = label['Name']
        if name == "License Plate":
            return label

def crop_image(img, source_name):
    im_bytes = img.read()

    image = Image.open(io.BytesIO(im_bytes))
    w, h = image.size

    response = client.detect_labels(Image={'Bytes': im_bytes})
    label = get_correct_label(response['Labels'])

    instance = label["Instances"][0]
    
    bbox = instance['BoundingBox']
    x0 = int(bbox['Left'] * w) 
    y0 = int(bbox['Top'] * h)
    x1 = x0 + int(bbox['Width'] * w)
    y1 = y0 + int(bbox['Height'] * h)

    img2 = image.crop((x0, y0, x1, y1))

    buf = io.BytesIO()

    img2.save(f"result_images/{source_name}") # dev
    img2.save(buf, format="JPEG")

    byte_img = buf.getvalue()
    return byte_img


def detect_text(img):
  res = client.detect_text(Image={'Bytes': img})

  textDetections = res['TextDetections']
  for text in textDetections:
    license_plate = text['DetectedText']
    if len(license_plate) == 7:
        return license_plate

def main():
    for i in range(6):
        read_plates(f"{i}.jpg")

def read_plates(file_name):
    with open(f"test_images/{file_name}", "rb") as image:
        cropped_image = crop_image(image, file_name)
    plate_number = detect_text(cropped_image)
    print(plate_number)

if __name__ == '__main__':
    main()