# Document Cleaning Feature Example

This example demonstrates how to use the new document cleaning feature in the RAG system.

## What is Document Cleaning?

Document cleaning is a preprocessing step removing unwanted content from documents before they are chunked and embedded. This is particularly useful when ingesting web content that contains:

- Advertisements
- Navigation menus
- Cookie notices
- Social media widgets
- Sidebar content
- Footer information
- Comments sections

## Basic Usage

### Clean a Web Page

```bash
# Basic cleaning with default settings
python main.py data-fill https://example.com/article --source-type url --enable-cleaning

# Clean with specific model
python main.py data-fill https://docs.python.org/3/tutorial/ --source-type url --enable-cleaning

# Clean multiple URLs
python main.py data-fill https://site1.com https://site2.com --source-type url --enable-cleaning
```

### Clean Local Files

Even local markdown files can benefit from cleaning if they contain unwanted metadata or formatting:

```bash
# Clean local markdown files
python main.py data-fill docs/*.md --enable-cleaning --mode elements
```

## Advanced Usage

### Custom Cleaning Prompt

You can provide a custom cleaning prompt for specialized content:

```bash
python main.py data-fill https://api-docs.example.com --source-type url --enable-cleaning --cleaning-prompt "Remove all navigation and marketing content. Keep only API documentation, code examples, and technical specifications. Preserve all code blocks and parameter descriptions."
```

### Environment Variables

Set up cleaning defaults using environment variables:

```bash
export RAG_ENABLE_CLEANING="true"

# Now cleaning will be enabled by default
python main.py data-fill https://technical-blog.com --source-type url
```

## Before and After Example

### Original Web Content

```html
<nav>Home | About | Products | Contact</nav>
<div class="ads">Buy our premium service!</div>
<article>
  <h1>How to Use Python Lists</h1>
  <p>Python lists are versatile data structures...</p>
  <pre><code>my_list = [1, 2, 3]</code></pre>
</article>
<aside>Related Articles: ...</aside>
<footer>Copyright 2024...</footer>
```

### After Cleaning

````markdown
# How to Use Python Lists

Python lists are versatile data structures...

```python
my_list = [1, 2, 3]
```
````

````

## Configuration Options

| Option | Description | Default | Environment Variable |
|--------|-------------|---------|---------------------|
| `--enable-cleaning` | Enable document cleaning | false | `RAG_ENABLE_CLEANING` |

## Best Practices

1. **Use cleaning for web content**: Web pages often contain lots of irrelevant content
2. **Monitor the logs**: Check reduction percentages to ensure cleaning is working properly
3. **Compare results**: Try with and without cleaning to see the difference in search quality

## Troubleshooting

### Common Issues

1. **No cleaning performed**: Check that `--enable-cleaning` is set
2. **API errors**: Ensure your API keys are set for cloud providers

### Debug Information

Enable debug logging to see detailed cleaning information:

```bash
python main.py data-fill https://example.com --source-type url --enable-cleaning --log-level DEBUG
````

This will show:

- Which documents are being cleaned
- Character count before/after cleaning
- Any errors during the cleaning process

## Performance Considerations

- Cleaning adds processing time proportional to document size and model speed
- Local models (Ollama) are slower but free
- Cloud models (OpenAI, Gemini) are faster but cost money
- Consider cleaning only when ingesting web content or messy documents
