# -*- coding: utf-8 -*-
import grpc
import service_pb2
import service_pb2_grpc
import threading

grpc_channel = grpc.insecure_channel('localhost:50051')
grpc_stub = service_pb2_grpc.CalStub(grpc_channel)


def run(n, m):
    response = grpc_stub.Add(service_pb2.AddRequest(number1=n, number2=m))  # 执行计算命令
    print(f"{n} + {m} = {response.number}")
    response = grpc_stub.Multiply(service_pb2.MultiplyRequest(number1=n, number2=m))
    print(f"{n} * {m} = {response.number}")


def handle(i):
    print('do task [{}]'.format(i))
    print('task finish')


def hello():
    with grpc.insecure_channel('10.0.50.153:50005') as channel:
        stub = service_pb2_grpc.LongServiceStub(channel)
        response = stub.hello(service_pb2.request(id='789', info='012'))
        for i in response:
            threading.Thread(target=handle, args=(i,)).start()


if __name__ == '__main__':
    # todo 参考方法
    hello()
    # todo 自定义方法
    i = 10
    j = 10
    run(n=i, m=j)
    # todo log
