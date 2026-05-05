from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import random, math, threading, time, os

# setting up the flask app, static_folder="." means it will look for html files in the same folder
app = Flask(__name__, static_folder=".")
# CORS lets the frontend talk to the backend even if they are on different ports, without this the browser blocks the requests
CORS(app)

# adjency list for network graph
# basically each key is a node and the list is who its connected to
# router and server are the main nodes, the pcs are the leaf nodes
network = {
    "router": ["server", "pc1", "pc2", "pc3"],
    "server": ["router", "pc4", "pc5", "pc6"],
    "pc1":    ["router", "pc2", "pc7"],
    "pc2":    ["router", "pc1", "pc8"],
    "pc3":    ["router", "pc9", "pc10"],
    "pc4":    ["server", "pc11"],
    "pc5":    ["server", "pc12"],
    "pc6":    ["server", "pc13"],
    "pc7":    ["pc1",   "pc14"],
    "pc8":    ["pc2",   "pc15"],
    "pc9":    ["pc3"],
    "pc10":   ["pc3"],
    "pc11":   ["pc4"],
    "pc12":   ["pc5"],
    "pc13":   ["pc6"],
    "pc14":   ["pc7"],
    "pc15":   ["pc8"],
}

# every node starts as "safe", we update this as attacks happen
node_status = {node: "safe" for node in network}

# 2. TRAFFIC SIMULATOR

# current attack being simulated, empty list means no attack
attack = []
# traffic log (not really used here but keeping it for now)
traffic = []
# set of currently infected nodes, using a set so we dont get duplicates
inodes = set()
# 30% chance malware will spread to a neighbor on each tick
spreadchance = 0.3

# IF ANY NODE IS ALREADY QURANTINED OR ISOLATED OR BLOCKED THEN WE SHOULD NOT USE IT
# we dont want to generate traffic from nodes that are already dealt with
def active_nodes():
    safe = ["pc1","pc2","pc3","pc4","pc5","pc6",
            "pc7","pc8","pc9","pc10","pc11",
            "pc12","pc13","pc14","pc15"]
    # filter out any node that has been actioned by the system
    active = [n for n in safe if node_status[n] not in ["QUARANTINE","ISOLATE","BLOCK"]]
    # if somehow all nodes are blocked just return the full list so the sim doesnt crash
    return active if active else safe

# these functions generate different types of traffic events with different characteristics
# each one returns a dict that represents one "packet event" in the simulation

# NORMAL TRAFFIC
# low packets per sec, medium size, common ports like 80 443 etc
def generate_normal():
    return {"src": random.choice(active_nodes()), "dst": random.choice(["server","router"]),
            "packets_per_sec": random.randint(1,10), "size": random.randint(200,800),
            "port": random.choice([80,224,443,8080]), "type": "normal"}

# ddos attacks have high packets per second
# tiny packet size because ddos just floods with small packets not big ones
# always targeting the server
def generate_ddos():
    return {"src": random.choice(active_nodes()), "dst": "server",
            "packets_per_sec": random.randint(500,1000), "size": random.randint(40,80),
            "port": 80, "type": "ddos"}

# port scan send packets to many ports but with low packets
# the unique_ports field is what makes this different from normal, its scanning a huge range
def generate_portscan():
    return {"src": random.choice(active_nodes()), "dst": random.choice(["server","router"]),
            "packets_per_sec": random.randint(50,100), "size": random.randint(40,60),
            "port": random.randint(100,500), "unique_ports": random.randint(100,500), "type": "portscan"}

# medium packets/sec but have large size bytes and unusual high ports.
# also the source can be from an infected node or a random one
# high ports like 4000-9000 are suspicious because normal apps dont use those
def generate_malware():
    # if we have infected nodes already, use one of them as the source (realistic behaviour)
    src = random.choice(list(inodes)) if inodes else random.choice(["pc1","pc2","pc3","pc4","pc5"])
    return {"src": src, "dst": random.choice(["server","router"]),
            "packets_per_sec": random.randint(20,40), "size": random.randint(1000,2000),
            "port": random.randint(4000,9000), "type": "malware"}

# this function will generate traffic and add chosen attacks randomly
# if no attack is set it just returns normal traffic
def get_traffic_event(attacks):
    if not attacks:
        return generate_normal()
    # pick a random attack from the list (there could be multiple types active)
    atk = random.choice(attacks)
    # map attack type string to the right generator function
    generators = {"ddos": generate_ddos, "portscan": generate_portscan, "malware": generate_malware}
    # if the attack type is not in the map it defaults to normal, just as a safety fallback
    return generators.get(atk["type"], generate_normal)()

# this function will simulate the spread of malware from the infected nodes to their neighbors based on a spreadchance which starts at 30% but can be adjusted by the RL agent based on the rewards.
# on each tick it tries to infect the neighbors which are not infected, quarantined, isolated or blocked.
def spread_malware():
    # collect new infections first then apply them all at once
    # if we added to inodes inside the loop it would mess up the iteration
    newi = set()
    for node in inodes:
        # skip nodes that are already contained, they cant spread
        if node_status[node] in ["QUARANTINE","ISOLATE","BLOCK"]:
            continue
        for n in network[node]:
            # only try to infect nodes that arent already infected and arent contained
            if n not in inodes and node_status[n] not in ["QUARANTINE","ISOLATE","BLOCK"]:
                # random.random() gives a float between 0 and 1, if its less than spreadchance we infect
                if random.random() < spreadchance:
                    newi.add(n)
                    node_status[n] = "infected"
    # now update the real infected set with all the new infections from this tick
    inodes.update(newi)

# ─────────────────────────────────────────
# K-MEANS
# ─────────────────────────────────────────

# standard euclidean distance formula, sqrt of sum of squared differences
# we need this to measure how far apart two traffic events are in 3D space (packets, size, port)
def euclidean_distance(p1, p2):
    return math.sqrt(sum((a-b)**2 for a,b in zip(p1,p2)))

# extract the 3 features we care about from an event dict into a plain list
# we only use packets_per_sec, size, port because these are the most distinctive features
def get_data(event):
    return [event["packets_per_sec"], event["size"], event["port"]]

# KMEANS START WITH 3 (because at max we are having two things: spread malware and any attack, so these will be two clusters and one will be normal traffic) RANDOM POINTS AS CENTER AND THEN ASSIGN EACH POINT TO ITS NEAREST CENTER AND THEN MOVE THE
# CENTER TO THE AVERAGE OF THEIR GROUP AND REPEAT THIS FOR 100 TIMES OR UNTIL CONVERGENCE
def kmeans(data, k):
    # pick k random points from the data as our starting centroids
    cent = random.sample(data, k)

    # run for 100 iterations, in practice it usually converges way before that
    for _ in range(100):
        # create k empty groups, one per centroid
        group = [[] for _ in range(k)]

        # assign every point to its closest centroid
        for point in data:
            closest = min(range(k), key=lambda i: euclidean_distance(point, cent[i]))
            group[closest].append(point)

        # recalculate each centroid as the mean of all points in its group
        newc = []
        for i, g in enumerate(group):
            if g:
                # average each dimension separately
                newc.append([sum(p[j] for p in g)/len(g) for j in range(3)])
            else:
                newc.append(cent[i])  # keep old center if group empty
        cent = newc

    return cent, group

# this function will run the kmeans on the last 20 traffic events and try to find anomalies by comparing the distance of the new event to the normal cluster and the attack cluster and also calculate a confidence score based on distance between the clusters.
def runkmeans(events):
    data = [get_data(e) for e in events]
    # need at least 3 points to form 3 clusters, otherwise skip
    if len(data) < 3:
        return None, 0

    cent, group = kmeans(data, 3)

    # the biggest cluster is probably normal traffic since most events are normal
    size = [len(g) for g in group]
    ncent_idx = size.index(max(size))
    ncent = cent[ncent_idx]

    # check where the most recent event lands, is it close to normal or close to an attack cluster?
    f = get_data(events[-1])
    d_normal = euclidean_distance(f, ncent)

    anomaly = None
    best_confidence = 0

    for i in range(len(group)):
        if i == ncent_idx or size[i] == 0:  # skip the normal cluster and empty clusters
            continue

        # confidence is based on how far the attack cluster is from the normal cluster
        # the further apart they are the more confident we are that its an attack
        dist = euclidean_distance(cent[i], ncent)
        confidence = min(100, int(dist / 10))  # cap at 100%

        d_attack = euclidean_distance(f, cent[i])

        # if the newest event is closer to the attack cluster than normal cluster, its suspicious
        if d_attack < d_normal and confidence > best_confidence:
            best_confidence = confidence
            anomaly = events[-1]  # newest packet is the anomaly

    return anomaly, best_confidence

# ─────────────────────────────────────────
# Swarm Intelligence
# ─────────────────────────────────────────

# this function is running swarm intelligence to trace the source of the attack
# its inspired by how ants leave pheromone trails, nodes that get visited more get higher marks
def runswarm(start):
    # initialize pheromone marks for all nodes to 0
    mark = {node: 0.0 for node in network}
    # give the starting node a high mark so agents are biased to start there
    mark[start] = 5.0

    # create 5 agents all starting from the suspected source node
    agents = [{"current": start, "visited": [start]} for _ in range(5)]

    # run for 15 steps
    for _ in range(15):
        for a in agents:
            curr = a["current"]
            # only consider neighbors we havent been to yet (no backtracking)
            unv = [n for n in network[curr] if n not in a["visited"]]
            if not unv:
                continue  # agent is stuck, just skip it

            # calculate probability for each neighbor based on its pheromone mark
            # +0.1 so no node ever has 0 probability (avoids division weirdness)
            scores = [mark[n] + 0.1 for n in unv]
            total = sum(scores)
            probs = [s/total for s in scores]

            # roulette wheel selection to pick the next node
            r = random.random()
            cum = 0
            chosen = unv[-1]  # default to last one in case of floating point issues
            for i, p in enumerate(probs):
                cum += p
                if r <= cum:
                    chosen = unv[i]
                    break

            # move the agent and leave a pheromone trail
            a["current"] = chosen
            a["visited"].append(chosen)
            mark[chosen] += 1.0

        # evaporate pheromones a bit each step so old paths fade out
        for node in mark:
            mark[node] *= 0.9  # pheromone evaporation

    # the node with the highest pheromone mark is our best guess for the attack source
    return max(mark, key=mark.get), mark

# ─────────────────────────────────────────
# decision tree
# ─────────────────────────────────────────

# this func looks how mixed the labels are in a group and returns 0 if all the labels are the same
# gini = 0 means perfect purity, gini = 0.5 means totally mixed (worst case for binary)
def gini(labels):
    if not labels:
        return 0
    total = len(labels)
    count = {}
    # count how many times each label appears
    for l in labels:
        count[l] = count.get(l, 0) + 1
    # gini formula: 1 - sum of (probability of each class)^2
    return 1 - sum((c/total)**2 for c in count.values())

# each node in the decision tree, either a decision node (has feature+threshold) or a leaf (has label)
class TNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature = feature       # which input to check (0=confidence, 1=attack_type, 2=spread)
        self.threshold = threshold   # what value to compare against
        self.left = left             # branch if value is below or equal to threshold
        self.right = right           # branch if value is above threshold
        self.label = label           # final answer (leaf node only)

# this function will try all the features and all the possible thresholds and return the one that gives the best score based on gini score; the lower the score the better the split
# this is basically brute force but it works fine for small datasets like ours
def best_split(data, labels):
    bestg, bestf, bestt = 999, None, None
    # try every feature
    for f in range(len(data[0])):
        # try every unique value in this feature as a threshold
        for row in data:
            t = row[f]
            # split data into left (<=t) and right (>t)
            ll = [labels[i] for i in range(len(data)) if data[i][f] <= t]
            rl = [labels[i] for i in range(len(data)) if data[i][f] > t]
            # skip this split if one side is empty, thats not a useful split
            if not ll or not rl:
                continue
            # weighted gini score for this split
            g = (len(ll)/len(labels))*gini(ll) + (len(rl)/len(labels))*gini(rl)
            if g < bestg:
                bestg, bestf, bestt = g, f, t
    return bestf, bestt

# this function will build the decision tree recursively
# it keeps splitting until all leaves are pure or we hit max depth
def build_tree(data, labels, depth=0, maxd=4):
    # base case 1: all labels are the same, make a leaf
    if len(set(labels)) == 1:
        return TNode(label=labels[0])
    # base case 2: hit max depth, just pick the most common label
    if depth == maxd:
        return TNode(label=max(set(labels), key=labels.count))

    f, t = best_split(data, labels)

    # base case 3: couldnt find a good split, just use majority label
    if f is None:
        return TNode(label=max(set(labels), key=labels.count))

    # split the data into left and right based on best feature and threshold
    ldata, ll, rdata, rl = [], [], [], []
    for i in range(len(data)):
        if data[i][f] <= t:
            ldata.append(data[i]); ll.append(labels[i])
        else:
            rdata.append(data[i]); rl.append(labels[i])

    # recursively build left and right subtrees
    return TNode(feature=f, threshold=t,
                 left=build_tree(ldata, ll, depth+1, maxd),
                 right=build_tree(rdata, rl, depth+1, maxd))

# this function will take a sample and traverse the tree based on features and thresholds and return the label of the leaf node it reaches at the end
def predict(tree, sample):
    # if we are at a leaf node return its label
    if tree.label is not None:
        return tree.label
    # otherwise go left or right depending on the threshold check
    return predict(tree.left, sample) if sample[tree.feature] <= tree.threshold else predict(tree.right, sample)

# we train the tree with these sample data
# format is [confidence, attack_type_encoded, spread_count]
# confidence is 0-100, attack type: 0=ddos, 1=portscan, 2=malware, spread is number of infected nodes
training_data = [
    [95,0,10],[88,0,8],[76,0,5],[90,0,12],   # ddos -> ISOLATE
    [70,1,2],[55,1,1],[65,1,3],[72,1,4],      # port_scan -> BLOCK
    [80,2,12],[72,2,9],[85,2,15],[78,2,11],   # malware -> QUARANTINE
    [25,0,1],[30,1,0],[20,2,1],[15,0,0],      # low confidence -> ALERT
]

# these are the labels for the training data which is used to train the decision tree
training_labels = [
    "ISOLATE","ISOLATE","ISOLATE","ISOLATE",
    "BLOCK","BLOCK","BLOCK","BLOCK",
    "QUARANTINE","QUARANTINE","QUARANTINE","QUARANTINE",
    "ALERT","ALERT","ALERT","ALERT"
]

# Builds the tree once at startup using the 16 training examples we defined above and then we retrain it after each decision with the new data we get from runkmeans() and runswarm() to improve its performance over time.
decision_tree = build_tree(training_data, training_labels)

# this function will add the new data to the training set and retrain the tree with the new data
# this is online learning basically, the tree gets smarter as the sim runs
def retrain_tree(confidence, attack_type, spread, action):
    # encode attack type as a number because the tree only works with numbers
    amap = {"ddos":0,"portscan":1,"malware":2,"normal":0,"port_scan":1}
    training_data.append([confidence, amap.get(attack_type,0), spread])
    training_labels.append(action)
    # rebuild the whole tree from scratch with the new data added
    return build_tree(training_data, training_labels)

# this will take the confidence score from kmeans, the attack type, the spread of attack and the source node and then it creates a sample and passes it to the decision tree to take action
def make_decision(confidence, attack_type, spread, source):
    global decision_tree
    # same encoding as retrain_tree, attack type needs to be a number
    attack_map = {"ddos":0,"port_scan":1,"malware":2,"normal":0,"portscan":1}
    sample = [confidence, attack_map.get(attack_type,0), spread]
    # let the tree decide what action to take
    action = predict(decision_tree, sample)
    # human readable explanation for the dashboard logs
    messages = {
        "ISOLATE":    f"{source} isolated — {attack_type} detected, {confidence}% confidence",
        "BLOCK":      f"Port blocked on {source} — {attack_type}, {confidence}% confidence",
        "QUARANTINE": f"{source} quarantined — {attack_type} spreading to {spread} nodes",
        "ALERT":      f"Alert on {source} — confidence {confidence}%, monitoring closely"
    }
    return {"action": action, "explanation": messages[action], "node": source}

# list to store all rewards so we can calculate cumulative score
reward_history = []

# this function evaluates the action taken by the tree and gives a reward based on how good the action is
# positive reward = good decision, negative = bad, 0 = neutral
def evaluate_action(action, ibefore, iafter):
    # quarantine worked if infection count went down
    if action == "QUARANTINE" and iafter < ibefore: return +1
    # isolate worked if no nodes are infected anymore
    if action == "ISOLATE" and iafter == 0: return +1
    # block worked if infection didnt grow
    if action == "BLOCK" and iafter <= ibefore: return +1
    # alert is bad if things got worse while we were just watching
    if action == "ALERT" and iafter > ibefore: return -1
    return 0  # neutral, nothing changed either way

# this function will update the malware spread chance based on reward history; if we are getting good rewards we increase the malware spread chance
# this is the RL part, good performance = harder difficulty, bad performance = easier
def update_spread_chance():
    global spreadchance
    score = sum(reward_history)
    if score > 3:
        spreadchance = 0.5  # doing well so make it harder
        label = "hard "
    elif score < 0:
        spreadchance = 0.1  # struggling so make it easier
        label = "easy "
    else:
        spreadchance = 0.3  # default difficulty
        label = "normal"
    print(f"[RL] Spread chance updated to {spreadchance:.0%} -- difficulty: {label}")

# ─────────────────────────────────────────
#  SIMULATION STATE (for the UI)
# ─────────────────────────────────────────
# this dict holds everything the frontend needs to know about the current state
# we just send the whole thing as JSON on every /state request

sim_state = {
    "tick": 0,           # which tick we are on (out of max_ticks)
    "max_ticks": 30,     # total number of ticks in the simulation
    "running": False,    # is the auto-run loop currently going
    "finished": False,   # did we complete all 30 ticks
    "attack_mode": "none",
    "logs": [],          # list of log messages to show in the UI
    "node_status": dict(node_status),   # copy of node_status so frontend can colour nodes
    "infected": [],      # list of currently infected node names
    "reward_history": [],
    "decisions": [],     # list of decision dicts so frontend can show a history table
    "traffic_counts": {"normal": 0, "ddos": 0, "portscan": 0, "malware": 0},  # for the bar chart
    "action_counts": {"ISOLATE": 0, "BLOCK": 0, "QUARANTINE": 0, "ALERT": 0}, # for action stats
    "last_event": None,  # most recent traffic event
    "last_anomaly": None,
    "swarm_source": None,  # what node swarm thinks the attack came from
}

trafficlog = []  # keeps last N events for kmeans to analyse
attacktypes = ["none", "ddos", "portscan", "malware"]
auto_thread = None  # reference to the background thread so we can stop it
sim_lock = threading.Lock()  # prevent race conditions between the auto thread and manual tick requests

# this is the main simulation step, everything happens in here
# called either manually (via /tick endpoint) or automatically by auto_run
def do_tick():
    global decision_tree, inodes, attack, spreadchance

    # lock so only one tick can run at a time, important when auto running
    with sim_lock:
        tick = sim_state["tick"]
        # check if we are done before doing anything
        if tick >= sim_state["max_ticks"] or sim_state["finished"]:
            sim_state["running"] = False
            sim_state["finished"] = True
            return

        sim_state["tick"] += 1
        t = sim_state["tick"]

        # helper to add a log entry with tick number and severity level
        def log(msg, level="info"):
            sim_state["logs"].append({"tick": t, "msg": msg, "level": level})

        # every 5 ticks we randomize the attack type to keep things interesting
        if (t-1) % 5 == 0:
            chosen = random.choice(attacktypes)
            attack = [] if chosen == "none" else [{"type": chosen}]
            # reset infected nodes when switching away from malware
            if chosen != "malware":
                inodes = set()
            sim_state["attack_mode"] = chosen
            # reset non-actioned nodes to safe so they can participate again
            node_status.update({n: "safe" for n in node_status if node_status[n] not in ["QUARANTINE","ISOLATE","BLOCK"]})
            log(f"Attack mode → {chosen.upper()}", "system")

        # if there are infected nodes try to spread malware this tick
        if inodes:
            spread_malware()
            log(f"Malware spreading — infected: {list(inodes)}", "warning")

        infected_before = len(inodes)  # snapshot before action so we can calculate reward later
        event = get_traffic_event(attack)
        trafficlog.append(event)
        sim_state["last_event"] = event
        # increment the counter for this traffic type for the chart
        sim_state["traffic_counts"][event["type"]] = sim_state["traffic_counts"].get(event["type"], 0) + 1
        log(f"Traffic: {event['type']} | {event['src']} → {event['dst']} | {event['packets_per_sec']} pkt/s", "traffic")

        # need at least 4 events before kmeans is worth running
        if len(trafficlog) < 4:
            sim_state["node_status"] = dict(node_status)
            sim_state["infected"] = list(inodes)
            return

        # run kmeans on the last 20 events to check for anomalies
        anomaly, confidence = runkmeans(trafficlog[-20:])
        sim_state["last_anomaly"] = anomaly

        # only act if we found an anomaly with high enough confidence and its not normal traffic
        if anomaly and confidence >= 50 and anomaly["type"] != "normal":
            log(f"K-Means anomaly! Confidence: {confidence}%", "alert")

            # use swarm to trace back to where the attack came from
            snode, pmap = runswarm(anomaly["src"])
            sim_state["swarm_source"] = snode
            spread = len(inodes)
            log(f"Swarm traced source → {snode} | spread: {spread}", "alert")

            # ask the decision tree what we should do
            dec = make_decision(confidence, anomaly["type"], spread, snode)
            node_status[snode] = dec["action"]  # apply the action to the node
            sim_state["action_counts"][dec["action"]] = sim_state["action_counts"].get(dec["action"], 0) + 1

            # save this decision to the history list for the UI table
            sim_state["decisions"].append({
                "tick": t,
                "action": dec["action"],
                "node": snode,
                "type": anomaly["type"],
                "confidence": confidence,
                "explanation": dec["explanation"]
            })
            log(f"Decision: {dec['action']} on {snode} — {dec['explanation']}", "decision")

            # if its malware and the source isnt already infected, add it now
            if anomaly["type"] == "malware" and anomaly["src"] not in inodes:
                inodes.add(anomaly["src"])
            # if we quarantined it, remove it from infected set
            if dec["action"] == "QUARANTINE":
                inodes.discard(snode)

            # retrain the tree with this new data point
            decision_tree = retrain_tree(confidence, anomaly["type"], spread, dec["action"])

            # calculate reward and save it
            iafter = len(inodes)
            reward = evaluate_action(dec["action"], infected_before, iafter)
            reward_history.append(reward)
            sim_state["reward_history"] = list(reward_history)
            log(f"Reward: {'+' if reward>=0 else ''}{reward} | Score: {sum(reward_history):+d}", "reward")
        else:
            log("K-Means: traffic normal.", "info")

        # always sync node_status and infected list back to sim_state at end of tick
        sim_state["node_status"] = dict(node_status)
        sim_state["infected"] = list(inodes)

        # check if we just finished the last tick
        if sim_state["tick"] >= sim_state["max_ticks"]:
            sim_state["finished"] = True
            sim_state["running"] = False
            log("=== Simulation Complete ===", "system")

# background thread that keeps calling do_tick every 1.5 seconds when auto running
def auto_run():
    while sim_state["running"] and not sim_state["finished"]:
        do_tick()
        time.sleep(1.5)  # wait between ticks so the UI has time to update

# ─────────────────────────────────────────
#  FLASK ROUTES
# ─────────────────────────────────────────

# serve the dashboard html file when someone opens the root url
@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")

# the frontend polls this every second to get the latest sim state
@app.route("/state")
def get_state():
    return jsonify(sim_state)

# lets the user step through the simulation one tick at a time manually
@app.route("/tick", methods=["POST"])
def manual_tick():
    if not sim_state["finished"]:
        do_tick()
    return jsonify({"ok": True})

# starts the auto run loop in a background thread
@app.route("/start", methods=["POST"])
def start_auto():
    global auto_thread
    # only start if its not already running and not finished
    if not sim_state["running"] and not sim_state["finished"]:
        sim_state["running"] = True
        # daemon=True means the thread dies when the main program exits
        auto_thread = threading.Thread(target=auto_run, daemon=True)
        auto_thread.start()
    return jsonify({"ok": True})

# pauses the auto run by setting running to False, the auto_run loop checks this flag
@app.route("/pause", methods=["POST"])
def pause_auto():
    sim_state["running"] = False
    return jsonify({"ok": True})

# resets everything back to the starting state so you can run the sim again
@app.route("/reset", methods=["POST"])
def reset_sim():
    global inodes, attack, trafficlog, decision_tree, reward_history, auto_thread
    with sim_lock:
        # reset all the sim_state fields to their defaults
        sim_state.update({
            "tick": 0, "max_ticks": 30, "running": False, "finished": False,
            "attack_mode": "none", "logs": [], "infected": [], "reward_history": [],
            "decisions": [], "traffic_counts": {"normal":0,"ddos":0,"portscan":0,"malware":0},
            "action_counts": {"ISOLATE":0,"BLOCK":0,"QUARANTINE":0,"ALERT":0},
            "last_event": None, "last_anomaly": None, "swarm_source": None,
        })
        # reset all global variables too
        inodes = set()
        attack = []
        trafficlog = []
        reward_history = []
        # set every node back to safe
        for n in node_status:
            node_status[n] = "safe"
        sim_state["node_status"] = dict(node_status)
        # rebuild the tree from only the original 16 training samples (not the ones added during the sim)
        decision_tree = build_tree(training_data[:16], training_labels[:16])
    return jsonify({"ok": True})

if __name__ == "__main__":
    print("Dashboard → http://localhost:5000")
    # debug=False so it doesnt restart every time a file changes (that would mess up the sim state)
    app.run(debug=False, port=5000)