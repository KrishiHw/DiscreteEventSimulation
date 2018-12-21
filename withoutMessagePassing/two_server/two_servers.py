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
processing = 50.00


def generator(_env, number, _s1_counter, _s2_counter):
    """Source generates requests randomly"""
    for i in range(number):
        s1 = server_one(_env, i, _s1_counter, _s2_counter)
        env.process(s1)
        arrival_ = random.expovariate(arrival)
        yield env.timeout(arrival_)


def server_one(_env, i, _s1_counter, _s2_counter):
    global s1_totalQueueLength
    global s1_sumWaitingTime

    arrival_time = env.now
    with _s1_counter.request() as req:
        yield req

        waiting_time = env.now - arrival_time
        processing_time = random.expovariate(processing)
        yield env.timeout(processing_time)

        s2 = server_two(_env, i, _s2_counter)
        env.process(s2)

        s1_totalQueueLength = s1_totalQueueLength + len(s1_counter.queue)
        s1_sumWaitingTime = s1_sumWaitingTime + waiting_time
        if i > 0:
            print("Server 1: average queue length = " + str(s1_totalQueueLength / i) + " average waiting time = " + str(
                s1_sumWaitingTime / i))

        file1.write(str(arrival_time) + "," + str(i) + "," + str(waiting_time) + "," + str(processing_time) + "," + str(
            len(_s1_counter.queue)) + "\n")


def server_two(_env, i, _s2_counter):
    global s2_totalQueueLength
    global s2_sumWaitingTime

    arrival_time = env.now

    with _s2_counter.request() as req:
        yield req
        waiting_time = env.now - arrival_time
        processing_time = random.expovariate(processing)
        yield env.timeout(processing)

        s2_totalQueueLength = s2_totalQueueLength + len(s2_counter.queue)
        s2_sumWaitingTime = s2_sumWaitingTime + waiting_time

        if i > 0:
            print("Server 2: average queue length = " + str(s2_totalQueueLength / i) + " average waiting time = " + str(
                s2_sumWaitingTime / i))

        file2.write(
            str(arrival_time) + "," + str(i) + "," + str(waiting_time) + "," + str(processing_time) + "," + str(
                len(_s2_counter.queue)) + "\n")


def get_pdf(data):
    a = np.array(data)
    ag = st.gaussian_kde(a)
    x = np.linspace(min(data), max(data), 1000)
    y = ag(x)
    return x, y


def get_data():
    s1_dataset = pandas.read_csv("two_server/s1_" + str(arrival) + ".csv")
    s2_dataset = pandas.read_csv("two_server/s2_" + str(arrival) + ".csv")

    s1_data = s1_dataset.iloc[:, 2]
    s2_data = s2_dataset.iloc[:, 2]

    return s1_data, s2_data


def get_plot():
    s1_data, s2_data = get_data()
    x_pdf1, y_pdf1 = get_pdf(s1_data)
    x_pdf2, y_pdf2 = get_pdf(s2_data)

    label = "AR=" + str(arrival) + "req/sec, PR= 50 req/sec"
    plt.figure()
    plt.plot(x_pdf1, y_pdf1, color='g', label="Server 1:" + label)
    plt.plot(x_pdf2, y_pdf2, color='orange', label="Server 2:" + label)
    plt.legend()
    plt.xlabel('Latency')
    plt.ylabel('Probability')
    output_image = "two_server/AR_" + str(arrival) + ".png"
    plt.savefig(output_image)
    plt.close()

    plt.figure()


if __name__ == '__main__':
    random.seed(RANDOM_SEED)

    s1_totalQueueLength = 0
    s1_sumWaitingTime = 0.0

    s2_totalQueueLength = 0
    s2_sumWaitingTime = 0.0

    columns = "Arrival Time, ID,Waiting_time_queue,Processing_Time,QueueLength"
    file1 = open("two_server/s1_" + str(arrival) + ".csv", "w")
    file1.write(columns + "\n")

    file2 = open("two_server/s2_" + str(arrival) + ".csv", "w")
    file2.write(columns + "\n")

    env = simpy.Environment()
    s1_counter = simpy.Resource(env, capacity=1)
    s2_counter = simpy.Resource(env, capacity=1)
    env.process(generator(env, NEW_REQUEST, s1_counter, s2_counter))
    env.run()

    get_plot()
