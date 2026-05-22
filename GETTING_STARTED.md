# FedCare - Getting Started Guide

Welcome to FedCare! This guide will help you get up and running in minutes.

## 🎯 What is FedCare?

FedCare is a federated learning system that enables hospitals to collaboratively train AI models **without sharing sensitive patient data**. Only model updates are shared, keeping patient information private.

**Architecture:**
- 🏥 **Edge Servers** (Hospitals) - Train locally on private data
- ☁️ **Main Server** (Cloud) - Aggregates model updates
- 🔒 **Privacy** - Raw data never leaves hospitals
- 📊 **Collaborative** - All hospitals benefit from shared learning

## ⚡ Quick Start (Choose One)

### Option 1: Local Python (Recommended for Learning)

**Time: 5 minutes**

```bash
# Step 1: Clone and setup
git clone https://github.com/ChiragPrasad2006/FedCare
cd FedCare

# Step 2: Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Start servers (run in separate terminals)
# Terminal 1:
python main_server/app.py

# Terminal 2:
HOSPITAL_ID=hospital_1 PORT=5001 python hospital_server/app.py

# Terminal 3:
HOSPITAL_ID=hospital_2 PORT=5002 python hospital_server/app.py

# Terminal 4:
HOSPITAL_ID=hospital_3 PORT=5003 python hospital_server/app.py

# Step 5: Run orchestrator (new terminal)
python orchestrator.py
```

**Expected Output:**
```
============================================================
Federated Learning for Healthcare - Starting...
============================================================

Initializing FedCare system...
Main server initialized
Configured hospital_1
Configured hospital_2
Configured hospital_3

Loading data into 3 hospital servers...
✓ hospital_1: 10000 samples loaded
✓ hospital_2: 10000 samples loaded
✓ hospital_3: 10000 samples loaded

Starting federated learning for 10 rounds...

ROUND 1
hospital_1 completed training - Loss: 0.4532, Accuracy: 0.8921
hospital_2 completed training - Loss: 0.4501, Accuracy: 0.8945
hospital_3 completed training - Loss: 0.4512, Accuracy: 0.8933
✓ Model aggregated - Round 1
...
```

---

### Option 2: Docker (Recommended for Production)

**Time: 3 minutes**

```bash
# Step 1: Clone
git clone https://github.com/ChiragPrasad2006/FedCare
cd FedCare

# Step 2: Start all services
docker-compose -f docker/docker-compose.yml up

# In another terminal, Step 3: Run orchestrator
python orchestrator.py
```

---

### Option 3: Kubernetes (Production Grade)

**Time: 10 minutes**

```bash
# Step 1: Clone
git clone https://github.com/ChiragPrasad2006/FedCare
cd FedCare

# Step 2: Deploy
bash deploy-k8s.sh

# Step 3: Check pods
kubectl get pods -n fedcare

# Step 4: Port forward
kubectl port-forward -n fedcare svc/main-server 5000:5000
kubectl port-forward -n fedcare svc/hospital-server 5001:5001

# Step 5: Run orchestrator
python orchestrator.py
```

---

## 🧪 Try Examples First

If you just want to see it in action:

```bash
cd FedCare
python examples.py
```

Choose from:
1. **Basic Flow** - Complete federated learning demo ✅ START HERE
2. **Non-IID Data** - Realistic heterogeneous data
3. **Direct API Calls** - REST API examples
4. **Orchestrator** - Full automation

---

## 📚 Documentation Map

Start with these based on your needs:

**🚀 I want to get started quickly**
→ Read this file + `docs/QUICKREF.md`

**📖 I want to understand the system**
→ Read `README.md` + `docs/ARCHITECTURE.md`

**🔧 I want to set it up properly**
→ Follow `docs/INSTALLATION.md`

**🔌 I want to use the APIs**
→ Check `docs/API.md`

**❓ I'm having issues**
→ See `docs/TROUBLESHOOTING.md`

---

## 🎓 How It Works

### Simple Explanation

```
Round 1:
1. Cloud sends model to hospitals
2. Hospital A trains on local data → updates model
3. Hospital B trains on local data → updates model
4. Hospital C trains on local data → updates model
5. Cloud averages all updates → new model
6. Repeat for Round 2, 3, etc.

Result: Better model without sharing patient data!
```

### Technical Flow

```
┌─────────────────────────────────────────┐
│  Initialize Global Model (Cloud)        │
└────────────┬────────────────────────────┘
             │
             ▼
    ┌────────────────────┐
    │ Distribute Model   │
    │ to Hospitals       │
    └────────────────────┘
             │
    ┌────────┼────────┐
    ▼        ▼        ▼
  [H1]     [H2]     [H3]  ← Train locally
    │        │        │
    └────────┼────────┘
             ▼
    ┌────────────────────┐
    │ Aggregate Weights  │
    │ (Cloud)            │
    └────────────────────┘
             │
             ▼
         New Model (Better!)
```

---

## 🔍 Verify Installation

Check if everything is working:

```bash
# Check main server
curl http://localhost:5000/health

# Check hospital server
curl http://localhost:5001/health

# Get system status
curl http://localhost:5000/status

# View metrics
curl http://localhost:5000/metrics | jq '.'
```

If all return JSON responses → ✅ **Everything works!**

---

## 🛠️ Common Tasks

### View Logs
```bash
tail -f logs/main_server_*.log
tail -f logs/hospital_server_*.log
```

### Check Training Progress
```bash
curl http://localhost:5000/metrics | jq '.metrics[-5:]'
```

### Reset System
```bash
curl -X POST http://localhost:5000/reset
```

### Stop Servers
```bash
# Local servers: Ctrl+C in each terminal
# Docker: docker-compose -f docker/docker-compose.yml down
# Kubernetes: kubectl delete namespace fedcare
```

---

## ⚙️ Customize Configuration

Edit `.env` to adjust:

```env
NUM_HOSPITALS=3              # More hospitals = more data
NUM_ROUNDS=10                # More rounds = better accuracy
EPOCHS_PER_ROUND=5           # More epochs = slower training
BATCH_SIZE=32                # Larger batch = faster but needs memory
LEARNING_RATE=0.001          # Lower = more stable, slower convergence
```

**Quick Presets:**

🚀 **Fast Testing** (runs in 2 min)
```env
NUM_HOSPITALS=2
NUM_ROUNDS=3
EPOCHS_PER_ROUND=2
BATCH_SIZE=64
```

🎯 **Balanced** (default settings)
```env
NUM_HOSPITALS=3
NUM_ROUNDS=10
EPOCHS_PER_ROUND=5
BATCH_SIZE=32
```

🏆 **Best Accuracy** (runs in 30+ min)
```env
NUM_HOSPITALS=5
NUM_ROUNDS=20
EPOCHS_PER_ROUND=10
BATCH_SIZE=16
```

---

## 🆘 Troubleshooting

### "Address already in use"
```bash
# Find and stop process using port
lsof -i :5000
kill -9 <PID>

# Or use different port
HOSPITAL_PORT=5010 python hospital_server/app.py
```

### "Connection refused"
```bash
# Make sure servers are running
ps aux | grep app.py
# If not, start them in separate terminals
```

### "Out of memory"
```env
# In .env
BATCH_SIZE=8          # Reduce batch size
NUM_HOSPITALS=2       # Reduce hospitals
```

### "Timeout"
```bash
# Servers might be slow, wait longer
# Check logs: tail -f logs/main_server_*.log
```

### More Issues?
See `docs/TROUBLESHOOTING.md` for 20+ solutions

---

## 📊 Understanding Results

After training, you'll see:

```
Round 1: Loss=0.4532, Accuracy=0.8921
Round 2: Loss=0.3421, Accuracy=0.9123  ← Loss decreased = good!
Round 3: Loss=0.2345, Accuracy=0.9312  ← Accuracy increased = good!
...
```

**What to look for:**
- ✅ Loss decreases over rounds
- ✅ Accuracy increases over rounds
- ✅ All hospitals reporting metrics
- ✅ Round times consistent

**If something's wrong:**
- ❌ Loss stays same → learning rate too low
- ❌ Loss increases → learning rate too high
- ❌ Hospital not reporting → check its logs
- ❌ Very slow → reduce batch size or epochs

---

## 🎯 Next Steps

### For Learning
1. ✅ Run the examples (`python examples.py`)
2. ✅ Read `README.md`
3. ✅ Try modifying data in `data_simulation/`
4. ✅ Check API docs in `docs/API.md`

### For Development
1. ✅ Understand architecture in `docs/ARCHITECTURE.md`
2. ✅ Modify models in `shared/models.py`
3. ✅ Add new data sources
4. ✅ Implement custom aggregation

### For Production
1. ✅ Follow `docs/INSTALLATION.md` (proper setup)
2. ✅ Use Docker or Kubernetes
3. ✅ Configure `.env` for your needs
4. ✅ Set up monitoring
5. ✅ Implement proper authentication

---

## 📞 Need Help?

1. **Quick Questions**: Check `docs/QUICKREF.md`
2. **Having Issues**: See `docs/TROUBLESHOOTING.md`
3. **Want to Learn**: Read `docs/ARCHITECTURE.md`
4. **API Questions**: Check `docs/API.md`
5. **Setup Issues**: Follow `docs/INSTALLATION.md`
6. **General Help**: See `README.md`

---

## 📁 Project Files

**Core System:**
- `main_server/app.py` - Cloud aggregator
- `hospital_server/app.py` - Edge trainer
- `shared/` - Common utilities
- `orchestrator.py` - Automation

**Deployment:**
- `docker/` - Docker containers
- `kubernetes/` - K8s setup
- Startup scripts - Easy launching

**Data & Examples:**
- `data_simulation/` - Data generation
- `examples.py` - Usage examples

**Documentation:**
- `README.md` - Full guide
- `docs/` - 5 detailed docs

Total: 27 files, 5500+ lines of code!

---

## ✨ Features

✅ Privacy-preserving ML
✅ Multi-hospital collaboration
✅ Decentralized data
✅ REST APIs
✅ Docker ready
✅ Kubernetes ready
✅ Comprehensive logging
✅ Error handling
✅ Multiple examples
✅ Extensive documentation

---

## 🎉 You're Ready!

You now have everything needed to:
- ✅ Run federated learning locally
- ✅ Deploy with Docker
- ✅ Scale with Kubernetes
- ✅ Understand the system
- ✅ Customize for your needs

**Start with:**
```bash
python examples.py
# Choose option 1
```

Then run:
```bash
python orchestrator.py
```

That's it! 🚀

---

## 📖 Full Documentation

- **README.md** - Complete overview (5000 words)
- **docs/INSTALLATION.md** - Setup guide (3000 words)
- **docs/API.md** - API reference (4000 words)
- **docs/ARCHITECTURE.md** - Technical design (5000 words)
- **docs/QUICKREF.md** - Quick reference (2000 words)
- **docs/TROUBLESHOOTING.md** - Problem solving (3000 words)

Total: 22,000+ words of documentation!

---

**Version**: 1.0
**Last Updated**: May 2024
**Status**: Production Ready ✅

Welcome aboard! 🎓
