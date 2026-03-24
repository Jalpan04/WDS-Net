import os
import shutil

def main():
    src_base = r"C:\Users\acer\.cache\kagglehub\datasets\hazrat\uhat-urdu-handwritten-text-dataset\versions\2\data\data"
    dst_base = r"d:\python projects\WSDNET+++\urdu_data"
    
    train_dst = os.path.join(dst_base, "Train")
    test_dst = os.path.join(dst_base, "Test")
    os.makedirs(train_dst, exist_ok=True)
    os.makedirs(test_dst, exist_ok=True)
    
    train_sources = [os.path.join(src_base, "characters_train_set"), os.path.join(src_base, "digits_train_set")]
    test_sources = [os.path.join(src_base, "characters_test_set"), os.path.join(src_base, "digits_test_set")]
    
    def copy_classes(src_dirs, dst_dir):
        count = 0
        for src in src_dirs:
            if not os.path.isdir(src): continue
            for cls in os.listdir(src):
                cls_path = os.path.join(src, cls)
                if not os.path.isdir(cls_path): continue
                dst_cls = os.path.join(dst_dir, cls)
                os.makedirs(dst_cls, exist_ok=True)
                for file in os.listdir(cls_path):
                    if file.endswith(('.jpg', '.png', '.jpeg')):
                        shutil.copy2(os.path.join(cls_path, file), os.path.join(dst_cls, file))
                        count += 1
        return count

    train_c = copy_classes(train_sources, train_dst)
    test_c = copy_classes(test_sources, test_dst)
    
    print(f"Copied {train_c} training images and {test_c} test images to {dst_base}")

if __name__ == '__main__':
    main()
