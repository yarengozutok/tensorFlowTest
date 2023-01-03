import tensorflow as tf

#kümeyi yükle
mnist = tf.keras.datasets.mnist #veri setini import ettik

#eğitim ve test olarak parçalıyoruz.
#eğitim verisi ile model kurulur ve test verisi ile model değerlendirilir

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)
 #veri setlerinin yapıları ekrana yazıldı

 #veri setini ölçekle 0-1
 #girdi verileri normalize edildi

x_train, x_test = x_train / 255.0, x_test / 255.0

#verideki bilgileri çıkaran bir filtre yapısı gibi düşün
#girdileri birleştirmek için flatten
#model oluşturuldu

model= tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation= "relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])
#model inşaa edildi

#sinir ağını eğitime hazır hale getirmek için modeli derle, optime et
#loss fonksiyonu  sinir ağının eğitim verisi üzerindeki performansını ölçer
# bunu sinir ağı ve gerçek değer karşılaştırması ile yapar
#sonuç değişkeni 10 kategoriden oluştuğu için bu loss fonksiyonunu kullanalım
#çıktı katmanında aktivasyon fonksiyonu kullanmadığım için from logits argumanına TRUE
#sinir ağının performansını görmek için metriks argümanı kullanılır
#analiz sınıflandırma olduğu için accuracy metriği
#modelin mimarisini inşa ettik ve compile metodu ile modeli derledik. model eğitime hazır

#model inşaa edildi
model.compile(
    optimizer= "adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

#eğitim verisi üzerindeki tüm verilerin bir kez sinir ağından geçmesi 1 epochs
#her bir epochdan sonra modelin kaybı doğruluğu ekrana yazılır.
#fit ile eğittik

#model eğitildi
model.fit(x_train,y_train,epochs=5)

#model eğitildikten sonra modelin test et üzerindeki performansını değerlendirmek için evaluate metodu kullanılır

#model test edildi
print("Evaulate yazılıyor...Test verisi üzerindeki doğruluk.")
model.evaluate(x_test, y_test)

#eğitim verisindeki doğruluk değeri ile test verisindekinin yakın olması istenir
#ilk sinir ağı inşaa edildi

while True:
  # kullanıcıdan el ile yazılan sayıyı alın
  digit = input("Please enter a handwritten digit: ")

  # sayıyı model için uygun hale getirin
  digit = digit.replace("\n", "")
  digit = digit.split(" ")
  digit = [int(x) for x in digit]
  digit = np.array(digit).reshape(1, 28, 28, 1)

  # modeli kullanarak tahmini alın
  prediction = model.predict(digit)

  # tahmini etikete dönüştürün
  label = np.argmax(prediction, axis=1)[0]

  # tahmini kullanıcıya gösterin
  print(f"Prediction: {label}")
