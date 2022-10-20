from tqdm import tqdm
import random

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Movie")
    args = parser.parse_args()

    name = f"./{args.dataset}_all.csv"
    test = open(f"./{args.dataset}_test", "w")
    train = open(f"./{args.dataset}_train", "w")
    val = open(f"./{args.dataset}_val", "w")
    r = random.random()
    random.seed(2022)
    with open(name, "r") as all:
        count = 0
        dataset = all.readlines()
        random.shuffle(dataset)
        dataset_len = len(dataset)
        testLen = int(dataset_len * 0.1)
        valLen = int(dataset_len * 0.2)
        for data in tqdm(dataset):
            count += 1
            if count <= testLen:
                test.write(data)
            elif count <= valLen:
                val.write(data)
            else:
                train.write(data)
    test.close()
    val.close()
    train.close()
