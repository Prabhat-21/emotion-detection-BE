from kafka import KafkaConsumer
from app import predict_emotion
def kafkaConsumer():
    bootstrap_servers = 'localhost:9092'
    topic_name = 'my-topic'
    consumer = KafkaConsumer(topic_name, bootstrap_servers=bootstrap_servers, auto_offset_reset='latest')
    consumer.subscribe(['my-topic'])
    for message in consumer:
        print(message.value)
        predict_emotion(message)

kafkaConsumer()


