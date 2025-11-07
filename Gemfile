# frozen_string_literal: true

source "https://rubygems.org"

# ======================================
# 🧱 CORE JEKYLL SETUP
# ======================================
gem "jekyll", "~> 4.3.3"

# ======================================
# 🎨 THEME (Sleek)
# ======================================
# Use the remote version of Sleek to ensure compatibility with Jekyll 4.x
gem "jekyll-remote-theme"

# ======================================
# ⚙️ PLUGINS
# ======================================
group :jekyll_plugins do
  gem "jekyll-seo-tag"         # SEO metadata
  gem "jekyll-sitemap"         # Sitemap.xml
  gem "jekyll-feed"            # Atom feed
  gem "jekyll-minifier"        # Minify HTML/CSS/JS
  gem "jekyll-redirect-from"   # Handle redirects / renamed URLs
  gem "jekyll-paginate"        # Optional: pagination support for blog posts
end

# ======================================
# 🧮 MARKDOWN + PARSER
# ======================================
gem "kramdown", "~> 2.4.0"
gem "kramdown-parser-gfm"      # GitHub-flavored markdown

# ======================================
# 🧰 DEVELOPMENT TOOLS
# ======================================
gem "webrick", "~> 1.8"        # Required for local serve in Ruby 3.x

# ======================================
# 🚀 DEPLOYMENT NOTES
# ======================================
# This setup is designed for GitHub Actions, not GitHub Pages’ limited environment.
# Remove 'github-pages' gem to unlock full control over Jekyll version and plugins.

gem "jekyll-paginate-v2", "~> 3.0"
