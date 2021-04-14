import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class ImageEnhancement:
    def denoise(self, img, type:str):
        if type == "Gaussian":
            return cv2.GaussianBlur(img, (7, 7), 0)
        if type == "NLMeans":
            return cv2.fastNlMeansDenoising(img, h=8, templateWindowSize=7, searchWindowSize=21)
        if type == "Bilateral":
            return cv2.bilateralFilter(img, -1, 5, 8)

    def moveLightSpot(self, img):
        img = img / 255.0
        thresh = np.mean(img) + 3 * np.std(img)
        #window 10*12
        h, w = img.shape
        for i in range(0, w):
            for j in range(0, h):
                #print(img[i:i+10, j:j+12])
                tmp = img[i:i+10, j:j+12]

                cnt = tmp[tmp >= thresh].size
                if cnt >20 or cnt == 0:
                    continue
                else:
                    tmp[tmp>thresh] = 1/(120-cnt) * np.sum(tmp[tmp<thresh])

        img = (img*255).astype(np.uint8)
        cv2.imwrite("removespot.jpg", img)
        return img


    def histoNormalize(self, img):
        dst = np.zeros_like(img)
        cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return dst

    def histoEqualize(self,img):
        #eq = cv2.equalizeHist(img)
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        eq = clahe.apply(img)
        return eq

    def piecewiseLinear(self, img, x1, y1, x2, y2):
        lut = np.zeros(256)
        for i in range(256):
            if i < x1:
                lut[i] = (y1 / x1) * i
            elif i < x2:
                lut[i] = ((y2 - y1) / (x2 - x1)) * (i - x1) + y1
            else:
                lut[i] = ((y2 - 255.0) / (x2 - 255.0)) * (i - x2) + y2
        out = cv2.LUT(img, lut)
        return np.uint8(out)

    def gammaCorrection(self, img):
        img = np.power(img/255.0, 0.7)
        img = (img*255).astype(np.uint8)
        return img

    def adaptiveThreshold(self, img, type:str):
        if type == "Gaussian":
            return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
        elif type == "Mean":
            return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)



class Morphology:
    def dilate(self, img):
        kernel = np.ones((3,3), dtype=np.uint8)
        out = cv2.dilate(img, kernel, 1)
        return out

    def erode(self, img):
        kernel = np.ones((3, 3), dtype=np.uint8)
        out = cv2.erode(img, kernel, 1)
        return out


class ImgAnalyze:
    def painHist(self, img, name):
        plt.hist(img.ravel(), 256, [0,256], rwidth=1.7)
        plt.title("gray scale histogram")
        plt.savefig(name + ".jpg")
        return


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    '''
    img = cv2.imread("testimg.jpg", cv2.IMREAD_GRAYSCALE)
    ime = ImageEnhancement()
    morph = Morphology()
    ima = ImgAnalyze()

    img[img>255] = 255
    img[img<0] = 0

    ime.moveLightSpot(img)

    img_dn = morph.erode(img)
    img_dn = morph.dilate(img_dn)
    #img_dn = ime.denoise(img_dn, "NLMeans")
    img_dn = ime.denoise(img_dn, "Bilateral")

    cv2.imwrite("denoise.jpg", img_dn)
    ima.painHist(img_dn, "denoise_hist")

    eq = ime.gammaCorrection(img_dn)
    #ima.painHist(eq, "gamma_hist")


    eq = ime.histoNormalize(eq)
    #ima.painHist(eq, "normal_hist")

    #eq = ime.histoEqualize(eq)
    ima.painHist(eq, "equal_hist")

    cv2.imwrite("eq.jpg", eq)
    '''
    ime = ImageEnhancement()
    morph = Morphology()
    ima = ImgAnalyze()

    pics = os.listdir("./img_woo/")
    for pic in pics:
        print(pic)
        img = cv2.imread("./img_woo/" + pic, cv2.IMREAD_GRAYSCALE)

        img[img > 255] = 255
        img[img < 0] = 0

        ime.moveLightSpot(img)

        img_dn = morph.erode(img)
        img_dn = morph.dilate(img_dn)
        # img_dn = ime.denoise(img_dn, "NLMeans")
        img_dn = ime.denoise(img_dn, "Bilateral")

        #cv2.imwrite("./data/" + pic, img_dn)
        #ima.painHist(img_dn, "denoise_hist")

        eq = ime.gammaCorrection(img_dn)
        # ima.painHist(eq, "gamma_hist")

        #eq = ime.histoNormalize(eq)
        # ima.painHist(eq, "normal_hist")

        eq = ime.histoEqualize(eq)
        #ima.painHist(eq, "equal_hist")

        cv2.imwrite("./data/withoutoval/" + pic, eq)








