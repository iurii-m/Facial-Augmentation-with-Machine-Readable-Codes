# -*- coding: utf-8 -*-
"""
testing the qr code generation

@author: iurii
"""
import qrcode
from PIL import Image

if __name__ == '__main__':
    img = qrcode.make("b'eNoBgAB//xE71DvssEK7SkCquMwy0rnBoBe02jWBOQE+ZL6QMMq/CL69vD485C2RPAM34TxvOJs/+TjPrwW+JD4pqR4927l/JJs5U7C4uv++F7dTu829gDhmNjW9yD3IOvA2PD1Hvpq1GbcctS086TlEvieveLw3PDU7iLzGvd054jhKPVg7dVA8iA=='")
    print(type(img), img)
    img.save("sample.png")

    # qr = qrcode.QRCode(
    #     version=1,
    #     error_correction=qrcode.constants.ERROR_CORRECT_H,
    #     box_size=10,
    #     border=4,
    # )
    
    # ERROR_CORRECT_L — About 7% or fewer errors can be corrected.
    # ERROR_CORRECT_M — About 15% or fewer errors can be corrected. This is the default value.
    # ERROR_CORRECT_Q — About 25% or fewer errors can be corrected.
    # ERROR_CORRECT_H — About 30% or fewer errors can be corrected.
    
    # qr.add_data('https://medium.com/@ngwaifoong92')
    # qr.make(fit=True)
    # img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
    # You can save it in as an image file as follow:
    # img.save("sample.png")