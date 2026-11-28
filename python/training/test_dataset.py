from dataset import CarinthiaDataset


dataset = CarinthiaDataset(
    csv_path="../../data/sample/data/carinthia.csv",
    dataset_root="../../data/sample",
)

print("전체 이미지 수:", len(dataset))

image, label = dataset[0]

print("첫 이미지 크기:", image.size)
print("첫 이미지 라벨:", label)