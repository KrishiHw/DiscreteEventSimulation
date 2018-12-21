import pandas
import simpy
import numpy as np
import scipy.stats as st
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

RANDOM_SEED = 42
NEW_REQUEST = 1000000

arrival = 25


def generator(_env, number, _s1_counter, _s2_counter, _s3_counter):
    """Source generates requests randomly"""
    for i in range(number):
        processing_time = random.expovariate(50.00)
        s1 = server_one(env, i, _s1_counter, _s2_counter, _s3_counter, processing_time)
        env.process(s1)
        arrival_ = random.expovariate(arrival)
        yield env.timeout(arrival_)


def server_one(_env, i, _s1_counter, _s2_counter, _s3_counter, processing_time):
    global s1_totalQueueLength
    global s1_sumWaitingTime
    global arrivalTime
    arrival_time = env.now

    with s1_counter.request() as req:
        yield req
        waiting_time = env.now - arrival_time
        yield env.timeout(processing_time)

        s2 = server_two(_env, i, _s2_counter, _s3_counter, processing_time)
        env.process(s2)

        s1_totalQueueLength = s1_totalQueueLength + len(s1_counter.queue)
        s1_sumWaitingTime = s1_sumWaitingTime + waiting_time
        if i > 0:
            print("Server 1: average queue length = " + str(s1_totalQueueLength / i) + " average waiting time = " + str(
                s1_sumWaitingTime / i))

        s1_file.write(
            str(arrival_time) + "," + str(i) + "," + str(waiting_time) + "," + str(processing_time) + "," + str(
                len(s1_counter.queue)) + "\n")


def server_two(_env, i, _s2_counter, _s3_counter, processing_time):
    global s2_totalQueueLength
    global s2_sumWaitingTime

    arrival_time = env.now

    with s2_counter.request() as req:
        yield req
        waiting_time = env.now - arrival_time
        yield env.timeout(processing_time)

        s3 = server_three(_env, i, _s3_counter, processing_time)
        env.process(s3)

        s2_totalQueueLength = s2_totalQueueLength + len(s2_counter.queue)
        s2_sumWaitingTime = s2_sumWaitingTime + waiting_time

        if i > 0:
            print("Server 2: average queue length = " + str(s2_totalQueueLength / i) + " average waiting time = " + str(
                s2_sumWaitingTime / i))

        s2_file.write(
            str(arrival_time) + "," + str(i) + "," + str(waiting_time) + "," + str(processing_time) + "," + str(
                len(s2_counter.queue)) + "\n")


def server_three(_env, i, _s3_counter, processing_time):
    global s3_totalQueueLength
    global s3_sumWaitingTime
    global endTime
    global arrivalTime
    global totalTime

    arrival_time = _env.now

    with _s3_counter.request() as req:
        yield req
        waiting_time = env.now - arrival_time
        yield _env.timeout(processing_time)
        endTime = env.now
        totalTime = endTime - arrivalTime

        s3_totalQueueLength = s3_totalQueueLength + len(s3_counter.queue)
        s3_sumWaitingTime = s3_sumWaitingTime + waiting_time

        if i > 0:
            print("Server 3: average queue length = " + str(s3_totalQueueLength / i) + " average waiting time = " + str(
                s3_sumWaitingTime / i))

        s3_file.write(
            str(arrival_time) + "," + str(i) + "," + str(waiting_time) + "," + str(processing_time) + "," + str(
                len(s2_counter.queue)) + "\n")


def get_pdf(data):
    a = np.array(data)
    ag = st.gaussian_kde(a)
    x = np.linspace(min(data), max(data), 1000)
    y = ag(x)
    return x, y


def get_data():
    s1_dataset = pandas.read_csv("messagePassing/s1_" + str(arrival) + ".csv")
    s2_dataset = pandas.read_csv("messagePassing/s2_" + str(arrival) + ".csv")
    s3_dataset = pandas.read_csv("messagePassing/s3_" + str(arrival) + ".csv")

    s1_data = s1_dataset.iloc[:, 2]
    s2_data = s2_dataset.iloc[:, 2]
    s3_data = s3_dataset.iloc[:, 2]

    return s1_data, s2_data, s3_data


def get_plot():
    s1_data, s2_data, s3_data = get_data()

    x_s1, y_s1 = get_pdf(s1_data)
    x_s2, y_s2 = get_pdf(s2_data)
    x_s3, y_s3 = get_pdf(s3_data)

    label = "AR=" + str(arrival) + "req/sec, PR= 50 req/sec"
    plt.figure()
    plt.plot(x_s1, y_s1, color='g', label="Server 1:" + label)
    plt.plot(x_s2, y_s2, color='orange', label="Server 2:" + label)
    plt.plot(x_s3, y_s3, color='blue', label="Server 3:" + label)
    plt.legend()
    plt.xlabel('Latency')
    plt.ylabel('Probability')
    output_image = "messagePassing/AR_" + str(arrival) + ".png"
    plt.savefig(output_image)
    plt.close()
    plt.figure()


if __name__ == '__main__':
    random.seed(RANDOM_SEED)

    arrivalTime = 0.0
    endTime = 0.0
    totalTime = 0.0

    s1_totalQueueLength = 0
    s1_sumWaitingTime = 0.0

    s2_totalQueueLength = 0
    s2_sumWaitingTime = 0.0

    s3_totalQueueLength = 0
    s3_sumWaitingTime = 0.0

    columns = "Arrival Time, ID,Waiting_time_queue,Processing_Time,QueueLength"

    s1_file = open("messagePassing/s1_" + str(arrival) + ".csv", "w")
    s1_file.write(columns + "\n")

    s2_file = open("messagePassing/s2_" + str(arrival) + ".csv", "w")
    s2_file.write(columns + "\n")

    s3_file = open("messagePassing/s3_" + str(arrival) + ".csv", "w")
    s3_file.write(columns + "\n")

    env = simpy.Environment()

    s1_counter = simpy.Resource(env, capacity=1)
    s2_counter = simpy.Resource(env, capacity=1)
    s3_counter = simpy.Resource(env, capacity=1)
    env.process(generator(env, NEW_REQUEST, s1_counter, s2_counter, s3_counter))
    env.run()

    get_plot()
