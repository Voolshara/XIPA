import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models

'''
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.compat.v1.Session() as sess:
    print(sess.run(c))
'''
# _________________________________________________________________________________________________Подготовка Дата сета
if len(tf.config.experimental.list_physical_devices('GPU')) != 0:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)  # __________Запуск с использованием видеоядра
BATCH_SIZE = 32  # Размер кластера
IMG_SIZE = (256, 256)  # Размер изображения (после реформации)

train_dataset = image_dataset_from_directory("train",
                                             shuffle=True,  # рандомизация данных
                                             batch_size=BATCH_SIZE,  # Подготовка сета для тренировки
                                             image_size=IMG_SIZE)

test_dataset = image_dataset_from_directory("test",
                                            shuffle=True,  # рандомизация данных
                                            batch_size=BATCH_SIZE,  # Подготовка сета для тестов
                                            image_size=IMG_SIZE)

class_names = train_dataset.class_names
'''
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))  # Примеры изображений c категориями
        plt.title(class_names[labels[i]])
        plt.axis("off")
'''
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),  # Искусственое увелечение дата сета
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

print('Number of train batches: %d' % tf.data.experimental.cardinality(test_dataset))  # Техническая инфа о
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))  # ___________________дата сете

# _________________________________________________________________________________________________Подготовка Дата сета

# _________________________________________________________________________________________________Обучение модели

epochs = 15
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # __________________________________Структура модели
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=40,
                    validation_data=test_dataset)
# _________________________________________________________________________________________________Обучение модели
# ___________________________________________________________________________________________________ Сохранение модели

model.save('Proj1_V2.h5')

# _________________________________________________________________________________________________ Описание модели
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')  # _____________________________________________________ Построение графика модели

test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print(test_acc)  # ___________________________________________________________________ Тестировние модели
plt.show()
