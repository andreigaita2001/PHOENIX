# 🔥 PHOENIX - Google Takeout Integration Guide

Complete guide to ingesting your 148GB of Google Takeout data into PHOENIX.

## 📋 Overview

This guide will help you:
1. Download your 148GB Google Takeout data from email
2. Extract and prepare the archives
3. Ingest everything into PHOENIX securely
4. Let your local AI learn about you from your data

**Everything stays local. Everything is encrypted. No cloud. No tracking.**

---

## 🚀 Quick Start (3 Steps)

### Step 1: Download Your Data

```bash
cd /home/user/PHOENIX
python download_google_takeout.py
```

Follow the prompts to:
- Download all Google Takeout files from your email
- Save them to `~/Downloads/GoogleTakeout/`
- Extract all archives automatically

### Step 2: Ingest Into PHOENIX

Once extraction is complete:

```bash
python ingest_takeout.py ~/GoogleTakeout_Extracted/Takeout
```

This will:
- ✅ Analyze your data structure
- ✅ Show you what will be processed
- ✅ Ask for confirmation
- ✅ Start ingesting with progress tracking
- ✅ Resume automatically if interrupted

### Step 3: Use Your Personalized AI

```bash
python phoenix.py
```

Your PHOENIX now knows you! Ask it things like:
- "What are my interests based on my YouTube history?"
- "When do I usually send emails?"
- "What patterns do you see in my data?"
- "What can you tell me about my habits?"

---

## 📊 What Gets Ingested

### Gmail
- **What:** Email metadata (sender, recipient, date, subject)
- **Privacy:** Full email content NOT stored (only metadata for patterns)
- **Insights:** Communication patterns, frequent contacts, peak email times

### Google Photos
- **What:** Photo metadata and file references
- **Privacy:** Photos stay in original location, only references encrypted
- **Insights:** Photo-taking habits, timeline of memories

### Location History
- **What:** GPS coordinates, timestamps
- **Privacy:** Encrypted daily summaries, exact coordinates protected
- **Insights:** Frequently visited places, travel patterns, home/work locations

### YouTube History
- **What:** Video titles, watch times, search history
- **Privacy:** Fully encrypted
- **Insights:** Interests, content preferences, viewing schedule

### Calendar
- **What:** Event titles, times, locations
- **Privacy:** Encrypted event data
- **Insights:** Schedule patterns, regular meetings, routines

### Contacts
- **What:** Names, emails, phone numbers
- **Privacy:** Encrypted contact information
- **Insights:** Relationship network, important people

### Google Drive
- **What:** Document metadata and file references
- **Privacy:** Original files untouched, only references stored
- **Insights:** Document organization, topics of interest

### Chrome
- **What:** Browsing history domains and titles
- **Privacy:** Encrypted URLs and domains only
- **Insights:** Web interests, research topics

### Maps
- **What:** Saved places, reviews, starred locations
- **Privacy:** Encrypted location data
- **Insights:** Favorite places, preferences

---

## 🔐 Privacy & Security

### Encryption
- **Algorithm:** AES-256 (Fernet)
- **Key Storage:** Local only, owner-only permissions (chmod 600)
- **Data Location:** `~/.phoenix_vault/` (local filesystem)

### What's Stored
- Encrypted metadata and patterns
- File references (not full content)
- Extracted insights and knowledge

### What's NOT Stored
- Full email bodies (privacy reasons)
- Original photo files (only references)
- Sensitive credential information

### Access Control
- Vault directory: `chmod 700` (owner only)
- Encryption key: `chmod 600` (owner read/write only)
- No network access during ingestion
- No telemetry or third-party access

---

## ⚡ Performance & Resources

### System Requirements
- **Disk Space:** 150GB+ free (for extraction + vault)
- **RAM:** 8GB minimum, 16GB recommended
- **CPU:** Any modern multi-core processor
- **Time:** 2-6 hours (depending on CPU speed)

### Optimization Tips
- Close other applications during ingestion
- Use SSD if available (much faster)
- Ensure stable power (use AC adapter for laptops)
- Don't interrupt the process (it will resume, but saves time)

### Resume Capability
If ingestion is interrupted:
```bash
python ingest_takeout.py ~/GoogleTakeout_Extracted/Takeout
```
It will automatically resume from where it left off!

Progress is saved to: `~/.phoenix_vault/ingestion_progress.json`

---

## 📈 Monitoring Progress

### During Ingestion
You'll see:
- Which service is being processed
- Number of items ingested
- Estimated time remaining
- Errors (if any)

### Logs
Full logs saved to: `~/.phoenix_vault/logs/ingestion_YYYYMMDD_HHMMSS.log`

View in real-time:
```bash
tail -f ~/.phoenix_vault/logs/ingestion_*.log
```

### Check Status
```bash
cat ~/.phoenix_vault/ingestion_progress.json
```

---

## 🧠 What PHOENIX Learns

After ingestion, PHOENIX will know:

### About You
- Your interests and hobbies
- Your communication style
- Your daily routines
- Your frequently visited places
- Your social network

### Patterns Detected
- Peak productivity hours
- Communication frequency
- Travel patterns
- Content preferences
- Schedule routines

### Insights Generated
- Personalized suggestions
- Habit identification
- Relationship mapping
- Interest clustering
- Productivity optimization tips

---

## 🎯 Using Your Personalized PHOENIX

### Query Your Data
```python
python phoenix.py
> what do you know about me?
> when am I most active?
> what are my top interests?
> show my communication patterns
```

### Search Your Vault
```python
> search my vault for "tennis"
> find emails about "project"
> show my calendar from last month
```

### Get Insights
```python
> analyze my habits
> suggest schedule optimizations
> what patterns do you see?
> give me personalized recommendations
```

---

## 🛠️ Troubleshooting

### Download Issues

**Problem:** Files won't download from email
**Solution:**
- Use direct download links from Google Takeout email
- Try different browser if downloads fail
- Check available disk space

**Problem:** Extraction fails
**Solution:**
- Ensure `unzip` is installed: `sudo apt install unzip`
- Try manual extraction with archive manager
- Check if archives are corrupted (re-download)

### Ingestion Issues

**Problem:** Out of memory error
**Solution:**
- Close other applications
- Process smaller batches (edit script to skip some services)
- Add swap space if needed

**Problem:** Permission denied
**Solution:**
```bash
chmod +x download_google_takeout.py
chmod +x ingest_takeout.py
```

**Problem:** Module not found
**Solution:**
```bash
pip install -r requirements.txt
```

### Performance Issues

**Problem:** Ingestion very slow
**Solution:**
- Check CPU usage (should be high during ingestion)
- Use SSD instead of HDD
- Ensure enough free disk space
- Close resource-heavy applications

---

## 🔄 Re-ingesting or Updating

### Add New Data
If you have new Google Takeout data:
```bash
python ingest_takeout.py ~/NewTakeout/Takeout
```
It will add new data without duplicating existing items.

### Start Fresh
To completely wipe and start over:
```python
from modules.personal_data_vault import PersonalDataVault
vault = PersonalDataVault()
vault.wipe_all_data(confirm="DELETE_ALL_MY_DATA")
```
⚠️ **WARNING:** This permanently deletes all ingested data!

---

## 📞 Need Help?

### Check Logs
```bash
ls -lh ~/.phoenix_vault/logs/
cat ~/.phoenix_vault/logs/ingestion_*.log
```

### View Progress
```bash
cat ~/.phoenix_vault/ingestion_progress.json | python -m json.tool
```

### Vault Statistics
```python
python -c "
from modules.personal_data_vault import PersonalDataVault
vault = PersonalDataVault()
print(vault.get_privacy_report())
"
```

---

## 🎉 Next Steps

After successful ingestion:

1. **Explore Your Data**
   - Ask PHOENIX questions about your patterns
   - Search through your vault
   - Review generated insights

2. **Customize Analysis**
   - Adjust pattern detection thresholds
   - Add custom categories
   - Train on specific interests

3. **Automate Tasks**
   - Set up automated insights generation
   - Create custom queries
   - Build personal dashboards

4. **Expand PHOENIX**
   - Add more data sources
   - Build custom modules
   - Integrate with other tools

---

## 🔒 Privacy Guarantee

**PHOENIX Promise:**
- ✅ All data stays on your machine
- ✅ No cloud uploads ever
- ✅ No telemetry or tracking
- ✅ Encrypted at rest
- ✅ Owner-only access
- ✅ Open source and auditable
- ✅ You have complete control

**Your data. Your AI. Your privacy.**

---

## 📚 Additional Resources

- Main README: `README.md`
- Test scripts: `test_personal_data_ingestion.py`
- Module docs: `modules/personal_data_vault.py`
- Privacy report: Run `python phoenix.py` and ask "show privacy report"

---

**Ready to give your AI a memory? Let's get started!**

```bash
python download_google_takeout.py
```
