import os
import urllib.request
import ssl

def download_file(url, target_path):
    print(f"Downloading {url} to {target_path}...")
    try:
        # Bypass SSL verification for simplicity in scripts (optional, depending on env)
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=context) as response, open(target_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print("Done.")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def main():
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../person_beauty_plugin/models"))
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    print(f"Target directory: {models_dir}")

    models = [
        {
            "url": "https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/models/onnx/version-RFB-320.onnx",
            "name": "face_detector.onnx"
        },
        {
            "url": "https://github.com/yakhyo/face-parsing/releases/download/v0.0.2/resnet18.onnx",
            "name": "face_parsing.onnx"
        },
        {
            # PFLD 68 Points (from cunjian/pytorch_face_landmark)
            "url": "https://raw.githubusercontent.com/cunjian/pytorch_face_landmark/master/onnx/pfld.onnx",
            "name": "face_landmark.onnx"
        }
    ]

    for model in models:
        target_path = os.path.join(models_dir, model["name"])
        if os.path.exists(target_path):
            print(f"{model['name']} already exists. Skipping.")
        else:
            download_file(model["url"], target_path)

if __name__ == "__main__":
    main()
