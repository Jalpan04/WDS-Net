import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    dataset_path = r"C:\Users\acer\.cache\kagglehub\datasets\gpreda\chinese-mnist\versions\7"
    csv_path = os.path.join(dataset_path, "chinese_mnist.csv")
    img_dir = os.path.join(dataset_path, "data", "data")
    
    out_dir = r"d:\python projects\chinese_mnist_dataset"
    train_dir = os.path.join(out_dir, "Train")
    test_dir = os.path.join(out_dir, "Test")
    
    df = pd.read_csv(csv_path)
    
    for c in df['value'].unique():
        os.makedirs(os.path.join(train_dir, str(c)), exist_ok=True)
        os.makedirs(os.path.join(test_dir, str(c)), exist_ok=True)
    
    # Stratified split to ensure 80/20 per class
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['value'], random_state=42)
    
    def copy_files(dataframe, dest_dir):
        count = 0
        for _, row in dataframe.iterrows():
            filename = f"input_{row['suite_id']}_{row['sample_id']}_{row['code']}.jpg"
            src = os.path.join(img_dir, filename)
            # using 'value' for the folder name
            dst = os.path.join(dest_dir, str(row['value']), filename)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                count += 1
        return count
    
    print("Copying training files...")
    train_count = copy_files(train_df, train_dir)
    print("Copying testing files...")
    test_count = copy_files(test_df, test_dir)
    
    print(f"Prepared {train_count} training images and {test_count} testing images in {out_dir}")

if __name__ == '__main__':
    main()
