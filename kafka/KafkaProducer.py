import kafka
import json


def kafkaProducer():
    producer = kafka.KafkaProducer(bootstrap_servers='localhost:9092',
                               api_version=(0, 11, 5),
                               value_serializer=lambda x: json.dumps(x).encode('utf-8'))
    data = "/man.png"
    topic = "my-topic"
    record = producer.send(topic, data)
    producer.close()



