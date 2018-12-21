import pandas
import simpy
import numpy as np
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt
import random
matplotlib.use('Agg')

RANDOM_SEED = 42
NEW_REQUEST = 1000000

arrival = 25
processing = 50.00


def generator(env, number, counter):
    """Source generates requests randomly"""
    for i in range(number):
        s1 = server(env, i, counter)
        env.process(s1)
        t = random.expovariate(arrival)
        yield env.timeout(t)


def server(env, i, counter):
    global totalQueueLength
    global sumWaitingTime

    arrival_time = env.now

    with counter.request() as req:
        yield req
        waiting_time = env.now - arrival_time
        processing_time = random.expovariate(processing)
        yield env.timeout(processing_time)
        totalQueueLength = totalQueueLength + len(counter.queue)

        sumWaitingTime = sumWaitingTime + waiting_time
        if i > 0:
            print("average queue length = " + str(totalQueueLength/i) + " average waiting time = " + str(sumWaitingTime/i))

        file.write(str(arrival_time)+"," + str(i) + "," + str(waiting_time) + "," + str(processing_time) + "," + str(len(counter.queue)) + "\n")


def get_pdf(data):
    a = np.array(data)
    ag = st.gaussian_kde(a)
    x = np.linspace(min(data), max(data), 1000)
    y = ag(x)
    return x, y


def get_plot():
    dataset = pandas.read_csv("single_server/s1_" + str(arrival) + ".csv")
    data = dataset.iloc[:, 2]

    x_pdf, y_pdf = get_pdf(data)

    label = "AR=" + str(arrival) + "req/sec, PR= 50 req/sec"
    plt.figure()
    plt.plot(x_pdf, y_pdf, label=label)
    plt.legend()
    plt.xlabel('Latency')
    plt.ylabel('Probability')
    output_image = "single_server/AR_" + str(arrival) + ".png"
    plt.savefig(output_image)
    plt.close()

    plt.figure()


if __name__ == '__main__':
    random.seed(RANDOM_SEED)
    env = simpy.Environment()
    file = open("single_server/s1_" + str(arrival)+ ".csv", "w")
    file.write("Arrival Time, ID,Waiting_time_queue,Processing_Time,QueueLength" + "\n")

    counter = simpy.Resource(env, capacity=1)
    env.process(generator(env, NEW_REQUEST, counter))
    env.run()
    get_plot()
