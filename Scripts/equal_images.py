import cv2
from tkinter import Tk, filedialog


def check_pixels(image1, image2):
    if image1.shape != image2.shape:
        return False

    shape = image1.shape

    for x in range(shape[0]):
        for y in range(shape[1]):
            for chanel1, chanel2 in zip(image1[x, y], image2[x, y]):
                if chanel1 != chanel2:
                    return False

    return True


def main():
    file_name1 = filedialog.askopenfilename(initialdir='../Results/img/')
    image1 = cv2.imread(file_name1)
    
    file_name2 = filedialog.askopenfilename()
    image2 = cv2.imread(file_name2)

    if check_pixels(image1, image2):
        print('Images are equal')
    else:
        print("Images are different")


if __name__ == '__main__':
    main()
