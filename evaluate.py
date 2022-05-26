import tensorflow as tf
import os
from train import load_image

if __name__ == '__main__':

    test_root = '/home/taozhi/datasets/face'

    model = tf.keras.models.load_model('model/model.h5')

    tot = 0
    count = 0
    for file in os.listdir(test_root):
        if file.endswith('.jpg'):
            img = os.path.join(test_root, file)
            img = load_image(img)
            img = tf.expand_dims(img, 0)

            res = model.predict(img)[0]
            res = tf.argmax(res).numpy()
            print(res)
            if res==0:
                tot += 1
            count += 1
            

    acc = tot / count

    print(acc)

    

    

