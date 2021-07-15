import re

class ScriptParser:

    def __init__(self, filename, cpu = True):
        self.filename = filename
        self.cpu = cpu
        self.samples = []
        self.features = []
        self.programs = []
        self.base_time = None

        self.regex = {
            'source': r"^\s*([a-f0-9]+)\s+(.+)\+(0x[a-f0-9]+)\s+\((.+)\)",
            'sample': r"^\s*([a-zA-Z0-9.-_]+)\s+([0-9]+)\s+([0-9]+\.[0-9]+):\s+([0-9]+)\s+([a-zA-Z0-9_\-\.]+):",
            'sample_cpu': r"^\s*([a-zA-Z0-9.-_]+)\s+([0-9]+)\s+\[([0-9]+)\]\s([0-9]+\.[0-9]+):\s+([0-9]+)\s+([a-zA-Z0-9_\-\.]+):"
        }

    def parse_source(self, line):
        match = re.search(self.regex['source'], line)
        if match:
            return {
                'address': match.group(1),
                'name': match.group(2),
                'offset': match.group(3),
                'origin': match.group(4)
            }
        return None

    def parse_sample(self, line):
        match = re.search(self.regex['sample'], line)
        if match:
            if not self.base_time:
                self.base_time = float(match.group(3))
            return {
                'program': match.group(1),
                'pid': match.group(2),
                'cpu': 0,
                'time': float(match.group(3)) - self.base_time,
                'period': int(match.group(4)),
                'event': match.group(5),
                'T': 0.0,
                'power': 0.0,
                'sources': []
            }
        return None

    def parse_sample_cpu(self, line):
        match = re.search(self.regex['sample_cpu'], line)
        if match:
            if not self.base_time:
                self.base_time = float(match.group(4))
            return {
                'program': match.group(1),
                'pid': match.group(2),
                'cpu': int(match.group(3)),
                'time': float(match.group(4)) - self.base_time,
                'period': int(match.group(5)),
                'event': match.group(6),
                'T': 0.0,
                'power': 0.0,
                'sources': []
            }
        return None

    def parse_line(self, line):
        if self.cpu:
            res = self.parse_sample_cpu(line)
        else:
            res = self.parse_sample(line)

        if res == None:
            res = self.parse_source(line)
        return res

    def add_feature(self, ev_name):
        if not ev_name in self.features:
            self.features.append(ev_name)

    def add_program(self, ev_prog):
        if not ev_prog in self.programs:
            self.programs.append(ev_prog)

    def parse(self):
        file = open(self.filename)

        while(line := file.readline()):
            line = line.replace('\n', '')
            res = self.parse_line(line)
            if res:
                if 'program' in res.keys():
                    self.add_feature(res['event'])
                    self.add_program(res['program'])
                    self.samples.append(res)
                else:
                    self.samples[-1]['sources'].append(res)
        file.close()

if __name__ == "__main__":
    parser = ScriptParser("./data/examples/test.txt")
    parser.parse()
