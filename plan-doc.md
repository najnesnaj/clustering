# MkDocs Site Setup Plan: "opencode applied"

## Overview
This plan outlines the steps to create a comprehensive MkDocs documentation site using the Material theme, organizing all existing markdown files from the clustering project into a structured documentation portal with GitHub Pages deployment.

## Current Files Analysis
**Total markdown files:** 11
- **Project Overview (2):** README.md, DOCKER_DEMO_FINAL.md
- **Planning Documents (4):** planning_document.md, plan-docker.md, PROJECT_SETUP_DOCUMENTATION.md, IMPLEMENTATION_COMPLETE.md  
- **How-to Guides (4):** README_DOCKER.md, DATABASE_SETUP.md, NOTEBOOK_COMPATIBILITY.md, demo_report.md
- **Jekyll-specific (1):** _pages/index.md (will be replaced)

## Step 1: Install Dependencies
```bash
pip install mkdocs mkdocs-material
```

## Step 2: Initialize MkDocs Structure
```bash
mkdocs new .               # Creates mkdocs.yml + docs/ directory
```

## Step 3: Directory Structure Creation
```
docs/
├── index.md                  # Landing page (enhanced README)
├── overview/
│   ├── project-intro.md      # README.md content
│   └── docker-demo.md        # DOCKER_DEMO_FINAL.md content
├── planning/
│   ├── initial-plan.md       # planning_document.md content
│   ├── docker-plan.md        # plan-docker.md content
│   ├── setup-docs.md         # PROJECT_SETUP_DOCUMENTATION.md content
│   └── implementation.md     # IMPLEMENTATION_COMPLETE.md content
├── guides/
│   ├── docker-deployment.md  # README_DOCKER.md content
│   ├── database-setup.md     # DATABASE_SETUP.md content
│   ├── notebooks.md          # NOTEBOOK_COMPATIBILITY.md content
│   └── demo-results.md       # demo_report.md content
└── assets/                   # For images, CSS, etc.
```

## Step 4: MkDocs Configuration (mkdocs.yml)

```yaml
site_name: "opencode applied"
site_description: "Comprehensive documentation for the Stock Clustering Analysis project"
site_author: "Najmus Saqib"
repo_url: https://github.com/najnesnaj/clustering
edit_uri: edit/main/docs/

# GitHub Pages deployment
remote_branch: gh-pages
remote_name: origin

theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - navigation.top
    - search.suggest
    - search.highlight
    - content.code.copy
    - content.code.annotate
  palette:
    - scheme: default
      primary: blue
      accent: light blue
    - scheme: slate
      primary: blue
      accent: light blue
  font:
    text: Roboto
    code: Roboto Mono

plugins:
  - search
  - minify:
      minify_html: true

markdown_extensions:
  - admonition
  - codehilite
  - toc:
      permalink: true
  - tables
  - pymdownx.superfences
  - pymdownx.highlight
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.details

nav:
  - Home: index.md
  - Project Overview:
      - Introduction: overview/project-intro.md
      - Docker Demo: overview/docker-demo.md
  - Planning & Development:
      - Initial Plan: planning/initial-plan.md
      - Docker Planning: planning/docker-plan.md
      - Setup Documentation: planning/setup-docs.md
      - Implementation Complete: planning/implementation.md
  - Deployment & Usage:
      - Docker Deployment: guides/docker-deployment.md
      - Database Setup: guides/database-setup.md
      - Notebook Compatibility: guides/notebooks.md
      - Demo Results: guides/demo-results.md
```

## Step 5: File Content Migration Strategy

### 5.1 Create Enhanced Landing Page (docs/index.md)
- Combine best elements from README.md
- Add navigation overview
- Include quick start section
- Add site description and purpose

### 5.2 Organize Project Overview Files
- **overview/project-intro.md:** Copy README.md content, add navigation breadcrumbs
- **overview/docker-demo.md:** Copy DOCKER_DEMO_FINAL.md content

### 5.3 Organize Planning Documents  
- **planning/initial-plan.md:** planning_document.md → initial-plan.md
- **planning/docker-plan.md:** plan-docker.md → docker-plan.md  
- **planning/setup-docs.md:** PROJECT_SETUP_DOCUMENTATION.md → setup-docs.md
- **planning/implementation.md:** IMPLEMENTATION_COMPLETE.md → implementation.md

### 5.4 Organize How-to Guides
- **guides/docker-deployment.md:** README_DOCKER.md → docker-deployment.md
- **guides/database-setup.md:** DATABASE_SETUP.md → database-setup.md
- **guides/notebooks.md:** NOTEBOOK_COMPATIBILITY.md → notebooks.md
- **guides/demo-results.md:** demo_report.md → demo-results.md

## Step 6: Content Enhancement Tasks

### 6.1 Add Navigation Elements
- Add breadcrumbs to each page
- Include "Edit on GitHub" links
- Add table of contents where needed

### 6.2 Format Consistency
- Ensure all headings follow proper hierarchy
- Add proper code block formatting where needed
- Include admonitions for important notes

### 6.3 Cross-Reference Links
- Link between related documents
- Add internal navigation links
- Reference external resources appropriately

## Step 7: Build and Test Commands

```bash
# Build the site locally
mkdocs build

# Serve locally for testing
mkdocs serve

# Deploy to GitHub Pages
mkdocs gh-deploy
```

## Step 8: GitHub Actions Integration

Create `.github/workflows/mkdocs.yml`:
```yaml
name: Deploy MkDocs Site

on:
  push:
    branches: [ main ]
    paths:
      - 'docs/**'
      - 'mkdocs.yml'
  pull_request:
    branches: [ main ]
    paths:
      - 'docs/**'
      - 'mkdocs.yml'

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pages: write
      id-token: write

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Required for mkdocs gh-deploy
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install mkdocs mkdocs-material
      
      - name: Deploy to GitHub Pages
        if: github.ref == 'refs/heads/main'
        run: mkdocs gh-deploy --force
      
      - name: Build site (for PR validation)
        if: github.ref != 'refs/heads/main'
        run: mkdocs build
```

## Step 9: GitHub Pages Configuration

1. **Enable GitHub Pages** in repository settings:
   - Go to repository Settings → Pages
   - Source: Deploy from a branch
   - Branch: `gh-pages` / `(root)`

2. **Automatic URL:**
   - Site will be available at: `https://najmus-saqib.github.io/clustering/`

3. **One-time setup:**
   ```bash
   mkdocs gh-deploy --setup
   ```

## Step 10: Clean Up
- Remove original markdown files from root (if desired)
- Update any existing references to old file locations
- Ensure all internal links work correctly

## Step 11: Optional Enhancements

### 11.1 Custom CSS (docs/assets/styles.css)
```css
.md-content {
    max-width: 80em;
}
.md-typeset .admonition {
    border-left: 4px solid #2196f3;
}
```

### 11.2 Additional MkDocs Plugins
```bash
pip install mkdocs-git-revision-date-localized-plugin
pip install mkdocs-awesome-pages-plugin
```

### 11.3 Navigation Enhancements
- Add search suggestions
- Include version information
- Add social links in footer

## Expected Outcome
A professional, searchable documentation site with:
- **3 main sections** clearly separating content types
- **Material design** with dark/light mode support
- **Mobile-responsive** layout
- **Full-text search** with suggestions
- **Code syntax highlighting** for technical content
- **Navigation breadcrumbs** for easy orientation
- **Tabbed navigation** for main sections
- **Automatic GitHub Pages deployment** via GitHub Actions

## Benefits of This Structure
1. **Logical Organization:** Clear separation between overview, planning, and guides
2. **Scalability:** Easy to add new documents in appropriate sections
3. **User-Friendly:** Multiple navigation methods (tabs, sections, search)
4. **Professional Appearance:** Material theme with customization
5. **Maintainable:** Clear file naming and directory structure
6. **Automated Deployment:** Seamless integration with GitHub Pages
7. **Version Control:** All changes tracked in git history

## Deployment Benefits
✅ **Automatic Deployment** - Push to main branch triggers deployment  
✅ **Version Control** - All changes tracked in git history  
✅ **Free Hosting** - GitHub Pages provides free static site hosting  
✅ **Custom Domain Support** - Easy to add custom domains  
✅ **HTTPS Included** - Automatic SSL certificates  
✅ **Fast Performance** - CDN delivery through GitHub's infrastructure  
✅ **Branch Protection** - Can restrict deployments to main branch only  
✅ **Rollback Capability** - Easy to revert to previous deployments  

This setup will create a comprehensive documentation portal that effectively communicates the project's purpose, development process, and usage instructions to different types of users (developers, users, and contributors).
