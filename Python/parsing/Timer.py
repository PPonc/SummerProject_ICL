from ScriptParser import ScriptParser

class Timer:

    def __init__(self, parser, N_cpu = 4):
        self.samples = parser.samples
        self.times = []
        for i in range(N_cpu):
            self.times.append(dict())



    def set_deltas(self):
        for s in self.samples:
            ev_name = s['event']
            ev_cpu = s['cpu']
            ev_time = s['time']
            if not ev_name in self.times[ev_cpu].keys():
                self.times[ev_cpu][ev_name] = [0.0, ev_time]
            if ev_time == 0.0:
                self.samples.remove(s)
            else:
                base_time = self.times[ev_cpu][ev_name][1]
                if ev_time == base_time:
                    base_time = self.times[ev_cpu][ev_name][0]
                else:
                    self.times[ev_cpu][ev_name][0] = self.times[ev_cpu][ev_name][1]
                    self.times[ev_cpu][ev_name][1] = ev_time
                s['T'] = ev_time - base_time
        if self.samples[0]['T'] == 0.0:
            self.samples.pop(0)



if __name__ == "__main__":
    parser = ScriptParser("./data/210713_1409_measure.script")
    parser.parse()

    timer = Timer(parser)
    timer.set_deltas()

    for i,sample in enumerate(timer.samples):
        if sample['T'] == 0.0:
            print(f"error {i} {sample['time']} {sample['event']} {sample['program']}")
