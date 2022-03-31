import os
import pandas as pd


def main():

    reviews = []

    # train
    # pos
    file_names = os.listdir("./train/pos")
    for i in range(len(file_names)):
        file_names[i] = "./train/pos/" + file_names[i]

    # reviews = []

    for file_name in file_names:
        with open(file_name, 'r') as f:
            review = f.readline()
            reviews.append((review, 'positive'))

    # train
    # neg
    file_names = os.listdir("./train/neg")
    for i in range(len(file_names)):
        file_names[i] = "./train/neg/" + file_names[i]

    # reviews = []

    for file_name in file_names:
        with open(file_name, 'r') as f:
            review = f.readline()
            review = review.replace("<br />", '')
            reviews.append((review, 'negative'))

    df = pd.DataFrame(reviews, columns=['review', 'sentiment'])
    df.to_csv('train.csv', index=False)

    reviews = []

    # test
    # pos
    file_names = os.listdir("./test/pos")
    for i in range(len(file_names)):
        file_names[i] = "./test/pos/" + file_names[i]

    # reviews = []

    for file_name in file_names:
        with open(file_name, 'r') as f:
            review = f.readline()
            reviews.append((review, 'positive'))

    # test
    # neg
    file_names = os.listdir("./test/neg")
    for i in range(len(file_names)):
        file_names[i] = "./test/neg/" + file_names[i]

    # reviews = []

    for file_name in file_names:
        with open(file_name, 'r') as f:
            review = f.readline()
            review = review.replace("<br />", '')
            reviews.append((review, 'negative'))

    df = pd.DataFrame(reviews, columns=['review', 'sentiment'])
    df.to_csv('test.csv', index=False)


if __name__ == "__main__":
    main()
