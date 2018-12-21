import random

import simpy

SEED = 42
arrival_rate = 45
average_processing_time = 1 / 50


def packet_generator(env, number, out_pipe):
    for i in range(number):
        time_between_arrivals = random.expovariate(arrival_rate)
        yield env.timeout(time_between_arrivals)
        processing_time = random.expovariate(1 / average_processing_time)
        arrival_time = env.now
        d = {1: processing_time, 2: i, 3: arrival_time}
        out_pipe.put(d)


def server(env, in_pipe):
    global queue_size1
    global queue_wait1
    while True:
        request = yield in_pipe.get()
        processing_time = request[1]
        i = request[2]
        arrival_time = request[3]
        waiting_time = env.now - arrival_time
        queue_length = len(in_pipe.items)
        yield env.timeout(processing_time)

        queue_wait1 = queue_wait1 + waiting_time
        queue_size1 = queue_size1 + queue_length

        if i > 0:
            print("waiting time = " + str(queue_wait1 / i) + ", queue length = " + str(queue_size1 / i))


if __name__ == '__main__':
    random.seed(SEED)
    queue_wait1 = 0
    queue_size1 = 0
    requests = 1000000
    environment = simpy.Environment()
    pipe = simpy.Store(environment)
    environment.process(packet_generator(environment, requests, pipe))
    environment.process(server(environment, pipe))
    environment.run()
