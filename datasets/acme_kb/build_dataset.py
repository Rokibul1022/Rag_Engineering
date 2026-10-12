"""Builds the labeled retrieval dataset (corpus.jsonl + queries.jsonl).  Run:  python datasets/acme_kb/build_dataset.py

The corpus mixes 56 hand-written passages (8 domains) with 30 near-identical template passages (router models,
error codes, plans). Each query is labeled with the passage that answers it AND with a query TYPE, because the types are
what expose the different failure modes of sparse and dense retrieval:

  keyword     shares distinctive words with the passage          -> sparse (BM25) should do well
  paraphrase  same meaning, (almost) no shared content words      -> dense should win, BM25 fails (vocabulary mismatch)
  question    a natural-language question                         -> mixed
  typo        key words misspelled                                -> BM25 fails (token never matches), subword models cope
  identifier  an exact code / model number / name                 -> sparse wins, dense blurs near-identical passages
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
K, P, Q, T, I = "keyword", "paraphrase", "question", "typo", "identifier"

TOPICS = [
 # ---------------------------------------------------------------- cloud
 ("cloud", "Object storage lifecycle rules",
  "AcmeCloud object storage supports lifecycle rules that automatically move objects to the Archive tier after a configurable number of days. Archived objects cost about 80 percent less to store but take up to 12 hours to restore.",
  [(K, "lifecycle rules archive tier days"), (P, "automatically shift old files into cheaper cold storage"), (Q, "How long does it take to restore archived objects?")]),
 ("cloud", "Autoscaling",
  "Autoscaling groups add or remove virtual machines based on CPU utilization. The default policy scales out when average CPU stays above 70 percent for five minutes and scales in below 30 percent.",
  [(K, "autoscaling CPU 70 percent scale out"), (P, "servers multiply by themselves whenever the processor gets busy"), (Q, "When does the default policy add more machines?")]),
 ("cloud", "Backup retention",
  "Daily database snapshots are kept for 35 days by default. Point-in-time recovery lets you restore a database to any second within the last 7 days.",
  [(K, "point-in-time recovery restore any second"), (P, "rewind the data store to an earlier moment after someone deleted rows by mistake"), (T, "databse snapshoot retention days")]),
 ("cloud", "API rate limits",
  "The AcmeCloud API allows 600 requests per minute per project. Requests above the limit receive HTTP status 429 and should be retried with exponential backoff.",
  [(K, "HTTP 429 exponential backoff"), (P, "what happens when I send too many calls to the interface too quickly"), (Q, "How many requests per minute are allowed?")]),
 ("cloud", "Two-factor authentication",
  "Administrators can require two-factor authentication for every member of an organization. Supported second factors are authenticator apps and hardware security keys; SMS codes are not supported.",
  [(Q, "Does AcmeCloud accept text message codes for 2FA?"), (P, "force everyone to prove who they are with an extra step when signing in"), (T, "two-facter authentification hardware keys")]),
 ("cloud", "Private networking",
  "Virtual private networks isolate your resources from the public internet. Traffic between subnets stays on the internal backbone and is never exposed to external routers.",
  [(P, "keep my servers hidden from the outside world"), (K, "private network subnets internal backbone")]),
 ("cloud", "Log retention",
  "Application logs are retained for 30 days and can be exported to cold storage for long-term compliance. Log search supports filters on severity, service name and time range.",
  [(K, "log retention 30 days export compliance"), (P, "how can auditors still read old diagnostic messages next year"), (Q, "Which filters does log search support?")]),
 ("cloud", "Support plans",
  "Premium support guarantees a 15 minute response time for critical incidents, available around the clock. Basic support offers email help with a next-business-day response.",
  [(Q, "What is the response time for critical incidents?"), (P, "how fast will an engineer answer when production is down"), (K, "premium support 15 minute response")]),
 # ---------------------------------------------------------------- hr
 ("hr", "Annual leave policy",
  "Full-time employees receive 25 days of paid annual leave per year. Up to five unused days can be carried over into the following year.",
  [(K, "annual leave carry over five days"), (P, "how much paid time off do staff get each twelve months"), (Q, "How many vacation days do employees get?")]),
 ("hr", "Sick leave policy",
  "Employees may take up to 10 days of paid sick leave per year. A medical certificate is required for absences longer than three consecutive days.",
  [(K, "sick leave medical certificate three days"), (P, "when must I show proof of illness to my manager"), (Q, "Do I need a doctor's note for sick days?")]),
 ("hr", "Parental leave",
  "Primary caregivers receive 16 weeks of fully paid parental leave, and secondary caregivers receive 6 weeks. Leave can be taken any time within the first year after the birth or adoption.",
  [(K, "parental leave 16 weeks caregivers"), (P, "time away from the office after having a baby"), (Q, "How long is leave for the second parent?")]),
 ("hr", "Remote work policy",
  "Employees may work remotely up to three days a week with manager approval. Remote workers must be reachable during core hours from 10:00 to 15:00.",
  [(Q, "What are the core hours for remote workers?"), (P, "how often may I do my job from home"), (K, "remote work three days manager approval")]),
 ("hr", "Expense reimbursement",
  "Business expenses must be submitted within 30 days with itemized receipts. Meals are reimbursed up to 60 dollars per day while traveling.",
  [(K, "expense reimbursement itemized receipts 30 days"), (P, "claim back money I spent on a work trip"), (T, "reimbusement recipts meals per day")]),
 ("hr", "Performance reviews",
  "Performance reviews take place twice a year, in June and December. Each review combines a self-assessment, peer feedback and a manager rating on a five-point scale.",
  [(Q, "How often are performance reviews held?"), (P, "how are my colleagues and boss going to judge my work"), (K, "review self-assessment peer feedback five-point scale")]),
 ("hr", "Learning budget",
  "Every employee has an annual learning budget of 1,200 dollars for courses, books and conferences. Unused budget does not roll over.",
  [(K, "learning budget 1,200 courses conferences"), (P, "money set aside for training and education"), (Q, "Does the unused learning money carry over?")]),
 ("hr", "Onboarding",
  "New hires receive a laptop on their first day and complete security training within two weeks. A buddy from the team guides them through the first month.",
  [(P, "what happens when a new colleague starts at the company"), (K, "new hires laptop first day security training"), (Q, "Who helps newcomers during the first month?")]),
 # ---------------------------------------------------------------- travel
 ("travel", "Japan entry requirements",
  "Citizens of many countries can visit Japan for up to 90 days without a visa. A passport valid for the whole stay and proof of onward travel are required at the border.",
  [(Q, "Do I need a visa to visit Japan for 30 days?"), (P, "entry rules for tourists arriving in Tokyo"), (T, "pasport valid onward travel japan")]),
 ("travel", "Tokyo transit card",
  "The Suica card works on trains, subways and buses across Tokyo and can also pay in convenience stores. It can be topped up at any station machine.",
  [(K, "Suica card Tokyo trains buses"), (P, "contactless pass for public transport in the Japanese capital"), (Q, "Where can I add money to a Suica?")]),
 ("travel", "Altitude sickness",
  "Altitude sickness can start above 2,500 meters. Climbing slowly, drinking water and resting on arrival help the body adjust.",
  [(K, "altitude sickness 2,500 meters"), (P, "feeling dizzy and breathless high up in the mountains"), (Q, "How can I avoid mountain sickness?")]),
 ("travel", "Baggage allowance",
  "Economy passengers may check one bag up to 23 kilograms. Overweight luggage is charged 15 euros per extra kilogram.",
  [(K, "baggage allowance 23 kilograms economy"), (P, "how heavy may my suitcase be on a budget ticket"), (Q, "What is the fee for overweight luggage?")]),
 ("travel", "Travel insurance",
  "Travel insurance covers medical emergencies, cancelled flights and lost luggage. Pre-existing conditions are usually excluded unless declared in advance.",
  [(P, "protection if my holiday falls apart"), (K, "travel insurance cancelled flights lost luggage"), (Q, "Are pre-existing conditions covered?")]),
 ("travel", "Louvre visit",
  "The Louvre is closed on Tuesdays. Tickets must be booked online for a fixed time slot, and the Mona Lisa is in Room 711.",
  [(I, "Mona Lisa Room 711"), (Q, "Which day is the Louvre closed?"), (P, "reserving an entry time for the famous Paris art gallery")]),
 # ---------------------------------------------------------------- cooking
 ("cooking", "Sourdough starter",
  "A sourdough starter is fed equal parts flour and water every 12 hours at room temperature. It is ready to bake with when it doubles in size within six hours of feeding.",
  [(K, "sourdough starter feeding flour water"), (P, "keep a wild yeast culture alive for homemade bread"), (T, "sourdogh starter feding")]),
 ("cooking", "Cooking basmati rice",
  "Basmati rice should be rinsed until the water runs clear, then cooked with one and a half parts water for 12 minutes. Let it rest covered for 5 minutes before fluffing.",
  [(Q, "How much water do I need for basmati rice?"), (P, "get light grains that do not clump together"), (K, "basmati rinse rest covered 5 minutes")]),
 ("cooking", "Knife safety",
  "Keep kitchen knives sharp because dull blades slip and cause more injuries. Cut on a stable board and curl your fingertips away from the blade.",
  [(P, "avoid hurting my hand while chopping vegetables"), (K, "knives sharp dull blades fingertips board")]),
 ("cooking", "Leftover food safety",
  "Cooked leftovers should be refrigerated within two hours and eaten within four days. Reheat food until it reaches 74 degrees Celsius throughout.",
  [(Q, "How long can leftovers stay in the fridge?"), (K, "leftovers refrigerated four days reheat 74"), (P, "is yesterday's dinner still safe to eat")]),
 ("cooking", "Espresso recipe",
  "A good espresso uses 18 grams of ground coffee and takes about 27 seconds to produce 36 grams of liquid. Water temperature should be near 93 degrees Celsius.",
  [(K, "espresso 18 grams 27 seconds"), (P, "pulling a strong small shot of caffeine"), (Q, "What water temperature is best for espresso?")]),
 ("cooking", "Baking soda vs powder",
  "Baking soda needs an acidic ingredient such as buttermilk or lemon juice to make dough rise, while baking powder already contains its own acid. Using the wrong one can leave a metallic taste.",
  [(Q, "What is the difference between baking soda and baking powder?"), (P, "why does my cake taste of metal"), (K, "baking soda acidic buttermilk")]),
 # ---------------------------------------------------------------- wellness
 ("wellness", "Sleep hygiene",
  "Adults generally need seven to nine hours of sleep. Keeping a regular schedule and avoiding screens before bed improves sleep quality.",
  [(K, "sleep seven to nine hours regular schedule screens"), (P, "how to rest better at night"), (Q, "How many hours of sleep do adults need?")]),
 ("wellness", "Hydration",
  "A common guideline is about two liters of water per day, more during exercise or hot weather. Pale yellow urine is a simple sign of good hydration.",
  [(Q, "How much water should I drink daily?"), (P, "easy way to tell whether I am drinking enough fluids"), (K, "hydration two liters pale yellow")]),
 ("wellness", "Beginner running plan",
  "Beginners can prepare for a 5 kilometer run in eight weeks by alternating walking and running three times a week. Rest days prevent injury.",
  [(K, "5 kilometer beginner plan eight weeks"), (P, "go from the sofa to finishing a short race"), (Q, "How often should a beginner run each week?")]),
 ("wellness", "Stretching",
  "Dynamic stretching before exercise warms up muscles, while static stretching is best after workouts. Hold each static stretch for 30 seconds.",
  [(K, "dynamic stretching static stretching 30 seconds"), (P, "loosening up limbs prior to a workout versus afterwards")]),
 ("wellness", "Protein intake",
  "Active adults often aim for 1.2 to 2.0 grams of protein per kilogram of body weight. Good sources include eggs, beans, fish and dairy.",
  [(Q, "How much protein per kilo of body weight?"), (K, "protein 1.2 to 2.0 grams eggs beans fish"), (P, "nutrient needed to build muscle and where to find it")]),
 # ---------------------------------------------------------------- finance
 ("finance", "Savings account interest",
  "The Acme Savings account pays 3.8 percent annual interest, calculated daily and paid monthly. There is no minimum balance.",
  [(K, "savings interest 3.8 percent annual"), (P, "how much extra money does my deposit earn every year"), (Q, "Is there a minimum balance for the savings account?")]),
 ("finance", "Credit card late fees",
  "Late payments incur a fee of 25 dollars, and the penalty interest rate rises to 29.9 percent. Setting up automatic payments avoids missed due dates.",
  [(K, "late payment fee 25 dollars penalty rate"), (P, "what if I forget to settle my card bill on time"), (Q, "What is the penalty interest rate?")]),
 ("finance", "International wire transfer",
  "International wire transfers cost 30 dollars and usually arrive within two business days. A SWIFT code and the recipient's IBAN are required.",
  [(I, "SWIFT code IBAN transfer"), (Q, "What details do I need to send money abroad?"), (P, "cost of moving funds to a foreign bank")]),
 ("finance", "Budgeting rule",
  "The 50/30/20 rule suggests spending 50 percent of income on needs, 30 percent on wants and saving 20 percent.",
  [(I, "50/30/20 rule"), (P, "simple formula to divide my monthly salary"), (Q, "What share of income should go to savings?")]),
 ("finance", "Diversification",
  "Diversification spreads investments across many assets so that one poor performer does not ruin the portfolio. Index funds provide broad diversification at low cost.",
  [(P, "do not put all your eggs in one basket"), (K, "diversification index funds portfolio low cost"), (Q, "Why do investors spread money across assets?")]),
 ("finance", "Emergency fund",
  "An emergency fund should cover three to six months of living expenses and be kept in an easily accessible account.",
  [(Q, "How big should an emergency fund be?"), (P, "cash cushion for unexpected hardship"), (K, "emergency fund three to six months expenses")]),
 # ---------------------------------------------------------------- tech support
 ("tech", "Slow Wi-Fi",
  "If the Wi-Fi is slow, restart the router, move closer to it and check for interference from microwave ovens. Switching to the 5 GHz band reduces congestion.",
  [(K, "Wi-Fi slow restart router 5 GHz"), (P, "my internet connection at home is sluggish"), (T, "wifi slwo restart rooter")]),
 ("tech", "Phone battery care",
  "Lithium-ion batteries last longer when kept between 20 and 80 percent charge. Avoid leaving the phone in hot cars, since heat accelerates aging.",
  [(P, "make my smartphone power cell survive for years"), (K, "lithium-ion battery 20 and 80 percent hot cars"), (Q, "Is it bad to leave my phone in a hot car?")]),
 ("tech", "Password managers",
  "A password manager generates and stores a unique long password for each account. Protect it with a strong master password and two-factor authentication.",
  [(K, "password manager unique master password"), (P, "how can I keep track of many different login secrets"), (Q, "What protects a password vault?")]),
 ("tech", "3-2-1 backup rule",
  "The 3-2-1 backup rule keeps three copies of data on two different media with one copy stored off-site.",
  [(I, "3-2-1 backup rule"), (P, "how many duplicates of my files should exist and where"), (Q, "What does the 3-2-1 rule mean for backups?")]),
 ("tech", "Printer offline",
  "When a printer shows offline, check that it is powered on and connected to the same network, then remove and re-add it in system settings.",
  [(Q, "Why does my printer say offline?"), (K, "printer offline re-add system settings network"), (P, "the machine for paper output refuses to respond")]),
 ("tech", "Bluetooth pairing",
  "To pair Bluetooth headphones, hold the power button for five seconds until the light blinks blue, then select the device from the phone's Bluetooth menu.",
  [(P, "connect wireless earphones to my mobile"), (K, "pair Bluetooth headphones power button blinks blue"), (Q, "How do I put headphones in pairing mode?")]),
 # ---------------------------------------------------------------- science
 ("science", "Photosynthesis",
  "Plants convert sunlight, water and carbon dioxide into glucose and oxygen through photosynthesis. The process takes place in chloroplasts, which contain the pigment chlorophyll.",
  [(P, "how green leaves make food out of light"), (T, "photosyntesis chlorophil chloroplasts"), (Q, "Where in the cell does photosynthesis happen?")]),
 ("science", "Plate tectonics",
  "Earth's crust is divided into plates that move a few centimeters per year. Earthquakes and volcanoes cluster along the boundaries where plates collide or slide past each other.",
  [(P, "why do tremors happen mostly near the edges of continents"), (K, "plates boundaries earthquakes volcanoes collide"), (Q, "How fast do tectonic plates move?")]),
 ("science", "Water cycle",
  "Water evaporates from oceans, condenses into clouds and returns as precipitation. Rivers carry runoff back to the sea.",
  [(K, "water cycle evaporates condenses precipitation"), (P, "how does seawater end up as rain and flow back again")]),
 ("science", "Newton's second law",
  "Newton's second law states that force equals mass times acceleration. A heavier object therefore needs a larger force to speed up at the same rate.",
  [(K, "force equals mass times acceleration"), (P, "why is a loaded truck harder to get moving than an empty one"), (Q, "What does Newton's second law say?")]),
 ("science", "DNA structure",
  "DNA stores genetic information in a double helix made of four bases: adenine, thymine, guanine and cytosine. Adenine pairs with thymine and guanine pairs with cytosine.",
  [(K, "DNA bases adenine thymine guanine cytosine"), (P, "the molecule carrying hereditary instructions"), (Q, "Which base pairs with guanine?")]),
 ("science", "Black holes",
  "A black hole is a region where gravity is so strong that nothing, not even light, can escape beyond the event horizon.",
  [(P, "cosmic object that traps even beams of brightness"), (K, "black hole event horizon gravity"), (Q, "What is the event horizon?")]),
 ("science", "How vaccines work",
  "Vaccines train the immune system to recognize a pathogen by presenting a harmless piece or weakened form of it. If the real pathogen appears later, the body responds faster.",
  [(P, "how does the body learn to fight a germ before meeting it"), (K, "vaccines immune system pathogen weakened"), (Q, "Why does the body respond faster after vaccination?")]),
 ("science", "Overfitting",
  "Overfitting happens when a model memorizes training data and performs poorly on new data. Regularization and more training examples reduce it.",
  [(P, "my model is great on practice data but fails in the real world"), (K, "overfitting regularization training examples"), (Q, "How can I reduce overfitting?")]),
 ("science", "Text embeddings",
  "An embedding maps text to a vector so that similar meanings lie close together. Cosine similarity compares the angle between two vectors.",
  [(K, "embedding vector cosine similarity angle"), (P, "numbers that capture what a sentence means"), (Q, "What does cosine similarity compare?")]),
 ("science", "Inflation",
  "Inflation is the rate at which prices rise, reducing purchasing power. Central banks often raise interest rates to slow it down.",
  [(Q, "What do central banks do to slow inflation?"), (P, "why does my money buy less every year"), (K, "inflation prices purchasing power interest rates")]),
 ("science", "Greenhouse effect",
  "Greenhouse gases such as carbon dioxide and methane trap heat in the atmosphere. Burning fossil fuels is the largest source of added carbon dioxide.",
  [(P, "why is the planet getting warmer"), (K, "greenhouse gases carbon dioxide methane fossil fuels"), (Q, "What is the biggest source of added carbon dioxide?")]),
 ("science", "Moon phases",
  "The Moon completes a cycle of phases every 29.5 days as the sunlit portion visible from Earth changes. A full moon occurs when Earth lies between the Sun and Moon.",
  [(K, "moon phases 29.5 days full moon"), (P, "why does our satellite change shape in the night sky"), (Q, "How long is the lunar cycle?")]),
]

# ---------------------------------------------------------------- templated, near-identical passages (hard for dense retrieval)
ROUTERS = [("RT-2210", "5", 1200, 4), ("RT-2215", "5", 1200, 4), ("RT-3340", "6", 1800, 4), ("RT-3345", "6", 1800, 5),
           ("RT-4410", "6", 3000, 5), ("RT-4415", "6", 3000, 8), ("RT-5520", "6E", 5400, 8), ("RT-5525", "6E", 5400, 8),
           ("RT-6600", "7", 9300, 8), ("RT-6605", "7", 9300, 8), ("RT-7710", "7", 19000, 8), ("RT-7715", "7", 19000, 12)]
ERRORS = [("E1001", "the storage quota was exceeded", "delete old snapshots or buy more storage"),
          ("E1002", "the API key is invalid", "create a new key in the console"),
          ("E2040", "the request timed out", "retry after a few seconds"),
          ("E2041", "the domain name could not be resolved", "check the DNS settings"),
          ("E3007", "the TLS certificate has expired", "renew the certificate"),
          ("E3008", "the TLS certificate does not match the host name", "issue a certificate for the correct name"),
          ("E4012", "the payment was declined by the bank", "update the billing card or contact your bank"),
          ("E4013", "the saved card has expired", "add a new card in billing settings"),
          ("E5503", "the service is temporarily unavailable", "wait and check the status page"),
          ("E5504", "the region is under maintenance", "switch to another region"),
          ("E6001", "the file is larger than the upload limit", "split the file into smaller parts"),
          ("E6002", "the file type is not allowed", "convert the file to a supported format")]
PLANS = [("Starter", 9, "5 GB", 1), ("Team", 29, "100 GB", 10), ("Business", 79, "1 TB", 50), ("Scale", 199, "5 TB", 250)]

corpus, queries = [], []


def add_doc(domain, title, text):
    did = f"d{len(corpus) + 1:03d}"
    corpus.append({"id": did, "title": title, "text": text, "domain": domain})
    return did


def add_query(qtype, text, rel):
    queries.append({"id": f"q{len(queries) + 1:03d}", "query": text, "relevant": [rel], "type": qtype})


for domain, title, text, qs in TOPICS:
    did = add_doc(domain, title, text)
    for qtype, qtext in qs:
        add_query(qtype, qtext, did)

for model, gen, speed, ports in ROUTERS:
    did = add_doc("products", f"Router {model}", f"The Acme {model} router supports Wi-Fi {gen} with a maximum speed of {speed} Mbps and has {ports} Ethernet ports.")
    if model in ("RT-2215", "RT-3345", "RT-4415", "RT-5520", "RT-6605", "RT-7715"):
        add_query(I, f"How many Ethernet ports does the {model} have?", did)
    if model in ("RT-3340", "RT-5525", "RT-7710"):
        add_query(I, f"{model} maximum speed", did)

for code, meaning, fix in ERRORS:
    did = add_doc("errors", f"Error {code}", f"Error {code} means {meaning}. Resolution: {fix}.")
    if code in ("E1002", "E2041", "E3008", "E4012", "E5504", "E6002"):
        add_query(I, f"what does error code {code} mean", did)
    if code in ("E3007", "E4013", "E6001"):
        add_query(I, f"{code} how to fix", did)

for name, price, storage, users in PLANS:
    did = add_doc("products", f"{name} plan", f"The {name} plan costs {price} dollars per month and includes {storage} of storage for up to {users} users.")
    add_query(K, f"price of the {name} plan", did)

# ---------------------------------------------------------------- typo variants of every 3rd keyword query (deterministic, so the file is reproducible)
def misspell(text):
    ws = text.split()
    cand = sorted((i for i, w in enumerate(ws) if len(w) >= 6 and w.isalpha()), key=lambda i: -len(ws[i]))[:2]
    for n, i in enumerate(cand):
        w = ws[i]
        m = len(w) // 2
        ws[i] = w[:m] + w[m + 1:] if n == 0 else w[:m] + w[m + 1] + w[m] + w[m + 2:]     # drop a letter / swap two letters
    return " ".join(ws)


for q in [q for q in queries if q["type"] == K][::3]:
    t = misspell(q["query"])
    if t != q["query"]:
        add_query(T, t, q["relevant"][0])

# ---------------------------------------------------------------- self-check: is each query type really what it claims?
STOP = set("a an the and or but if of to in on at for with by from as is are was were be been it this that these those we our you your they their he she his her its not no so than then there which who what when where how why do does did has have had into over under about also can could should may might i my me us".split())
words = lambda s: [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]
by_id = {d["id"]: d for d in corpus}
overlap = {}
for q in queries:
    dw = set(words(by_id[q["relevant"][0]]["text"] + " " + by_id[q["relevant"][0]]["title"]))
    qw = words(q["query"])
    q["lexical_overlap"] = round(sum(w in dw for w in qw) / max(1, len(qw)), 2)
    overlap.setdefault(q["type"], []).append(q["lexical_overlap"])

if __name__ == "__main__":
    with open(HERE / "corpus.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in corpus)
    with open(HERE / "queries.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(q, ensure_ascii=False) + "\n" for q in queries)
    print(f"{len(corpus)} passages, {len(queries)} queries")
    for t, v in overlap.items():
        print(f"  {t:<11} n={len(v):<3} mean word overlap with the answer passage = {sum(v) / len(v):.2f}")
    bad = [q for q in queries if q["type"] == P and q["lexical_overlap"] > 0.34]
    print("paraphrase queries that share too many words (>0.34):", [(q["id"], q["lexical_overlap"], q["query"]) for q in bad])
