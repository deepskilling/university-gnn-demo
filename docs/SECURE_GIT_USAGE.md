# Secure Git Usage with Environment Variables

## 🔐 **Security Setup Complete!**

Your GitHub Personal Access Token is now securely stored in `.env` file.

## 📁 **Files Created:**
- ✅ `.env` - Contains your GitHub PAT (automatically ignored by Git)
- ✅ `git_with_env.sh` - Helper script for Git operations
- ✅ `.gitignore` - Already configured to ignore sensitive files

## 🚀 **How to Use:**

### **Method 1: Using the Helper Script**
```bash
# Show configuration (safe - hides full token)
./git_with_env.sh

# Quick push with default message
./git_with_env.sh push

# Push with custom message  
./git_with_env.sh push "Add new feature"
```

### **Method 2: Manual Commands**
```bash
# Load environment variables first
source .env

# Then use in git commands
git push https://$GITHUB_USERNAME:$GITHUB_TOKEN@github.com/$GITHUB_USERNAME/$GITHUB_REPO.git main
```

### **Method 3: Direct Command (for one-off operations)**
```bash
# Using the stored token directly (replace YOUR_PAT with actual token)
git push https://deepskilling:YOUR_GITHUB_PAT@github.com/deepskilling/university-gnn-demo.git main
```

## 🛡️ **Security Features:**

✅ **Token Hidden from Git**: `.env` file is in `.gitignore`  
✅ **Token Partially Masked**: Helper script shows only first 10 characters  
✅ **Local Only**: Environment file stays on your machine  
✅ **Easy Rotation**: Just update `.env` file when token expires  

## 📋 **Current Configuration:**
- **GitHub Username**: `deepskilling`
- **Repository**: `university-gnn-demo`  
- **Token**: `ghp_XXXXXX...` (securely stored in .env file)
- **Repository URL**: `https://github.com/deepskilling/university-gnn-demo`

## ⚠️ **Security Best Practices:**

1. **Never commit `.env` file** (already in `.gitignore`)
2. **Rotate tokens regularly** (every 3-6 months)  
3. **Use minimal permissions** (your token has repo access only)
4. **Keep tokens private** (don't share or post online)

## 🔄 **Future Updates:**

When you want to push changes to GitHub:

```bash
# Add your changes
git add .
git commit -m "Your commit message"

# Push using stored credentials
./git_with_env.sh push "Your commit message"
```

## 🎯 **Benefits:**

- **Secure**: No hardcoded tokens in scripts
- **Convenient**: Easy to use helper script  
- **Maintainable**: One place to update credentials
- **Safe**: Automatic credential protection
