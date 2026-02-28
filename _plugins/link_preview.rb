require 'uri'

module Jekyll
  class LinkPreviewTag < Liquid::Tag
    def initialize(tag_name, markup, tokens)
      super
      @url = markup.strip
    end

    def render(context)
      url = @url
      
      begin
        uri = URI.parse(url)
        domain = uri.host
        
        html = <<-HTML
<div class="link-preview-card">
  <a href="#{url}" target="_blank" rel="noopener noreferrer" class="link-preview-link">
    <div class="link-preview-image">
      <img src="https://www.google.com/s2/favicons?domain=#{domain}&sz=128" alt="#{domain}" loading="lazy" />
    </div>
    <div class="link-preview-content">
      <div class="link-preview-title">#{domain}</div>
      <div class="link-preview-description">Click to visit this website</div>
      <div class="link-preview-domain">#{domain}</div>
    </div>
  </a>
</div>
        HTML
        
        html
      rescue => e
        puts "Error generating link preview for #{url}: #{e.message}"
        return nil
      end
    end
  end
end

Liquid::Template.register_tag('link_preview', Jekyll::LinkPreviewTag)
