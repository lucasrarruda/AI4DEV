import cv2
import pytesseract
from pytesseract import Output
import re

def extract_aws_components(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or invalid image path.")

    # Preprocess image (convert to gray and threshold)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_OTSU)

    # Show the thresholded image for debugging
    cv2.imshow("Thresholded Image", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # OCR extraction
    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(thresh, output_type=Output.DICT, config=custom_config)

    # AWS common components keywords (expand as needed)
    aws_keywords = [
        "EC2", "S3", "Lambda", "RDS", "DynamoDB", "VPC", "CloudFront", "IAM", "ECS", "EKS",
        "SNS", "SQS", "CloudWatch", "API Gateway", "Route 53", "Elastic Beanstalk", "Redshift"
    ]
    aws_keywords_regex = re.compile(r'\b(' + '|'.join(aws_keywords) + r')\b', re.IGNORECASE)

    # Extract detected words and filter AWS components
    components = set()
    for word in data['text']:
        matches = aws_keywords_regex.findall(word)
        for match in matches:
            components.add(match.upper())

    return list(components)

if __name__ == "__main__":
    image_path = "C:\\Projects\\AI4DEV\Hackaton\\aws_sample_architecture2.png"
    components = extract_aws_components(image_path)
    print("Detected AWS Components:")
    for comp in components:
        print("-", comp)