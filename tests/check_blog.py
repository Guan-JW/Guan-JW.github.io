#!/usr/bin/env python3
"""Build the real site in isolated copies and check the publishing boundary."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (ROOT / '_config.yml').read_text()
SWITCH = 'blog_enabled             : &blog_enabled false'
assert SWITCH in CONFIG, 'Run checks with the online blog switch off.'


def read(path):
    return path.read_text(encoding='utf-8')


with tempfile.TemporaryDirectory(prefix='guan-blog-check-') as tmp:
    source = Path(tmp).resolve() / 'source'
    destination = Path(tmp).resolve() / 'output'
    def ignore(path, names):
        skipped = {name for name in names if name.startswith('_site') or
                   name in ('.git', '.bundle', 'node_modules', '__pycache__')}
        if Path(path) == ROOT:
            skipped.add('vendor')
        return skipped
    shutil.copytree(ROOT, source, ignore=ignore)
    for name, published, date in [
        ('ready-sentinel', 'true', '2020-01-02'),
        ('draft-sentinel', 'false', '2020-01-03'),
        ('future-sentinel', 'true', '2199-01-01'),
    ]:
        (source / '_blog' / (name + '.md')).write_text(
            f'---\ntitle: {name}\ndate: {date}\npublished: {published}\n'
            f'tags: [sentinel-topic]\nexcerpt: {name}-summary\n---\n'
            f'## A heading\n\n{name}-body\n', encoding='utf-8')
    env = dict(os.environ, BUNDLE_GEMFILE=str(ROOT / 'Gemfile'), JEKYLL_ENV='production')

    def build(enabled=False, preview=False, configured=False, interactions=True):
        config = CONFIG.replace(SWITCH, SWITCH.replace('false', 'true')) if enabled else CONFIG
        if configured:
            config = config.replace('server_url: ""', 'server_url: "https://comments.example.test"')
        if not interactions:
            config = config.replace('blog_interactions:\n  enabled: true', 'blog_interactions:\n  enabled: false')
        (source / '_config.yml').write_text(
            config)
        configs = [source / '_config.yml']
        if preview:
            configs += [source / '_config.dev.yml', source / '_config.blog-preview.yml']
        result = subprocess.run([
            'bundle', 'exec', 'jekyll', 'build', '--safe', '--source', str(source),
            '--destination', str(destination), '--config', ','.join(map(str, configs)),
        ], cwd=ROOT, env=env, text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        for filename in ['feed.xml', 'sitemap.xml']:
            ET.parse(destination / filename)
        assert not (destination / 'services').exists(), 'Server source leaked into the website'

    def assert_no_mentions(*needles):
        for file in destination.rglob('*'):
            if file.suffix in ('.html', '.xml', '.json'):
                text = read(file)
                for needle in needles:
                    assert needle not in text, f'{needle} leaked into {file}'

    def assert_off():
        home = read(destination / 'index.html')
        assert 'Recent Posts' not in home
        assert '>Blog</a>' not in home
        for section in ['About Me', 'Publications', 'Honors and Awards', 'Education']:
            assert section in home, section
        for route in ['blog', 'tags', 'categories', 'year-archive', 'wordpress/blog-posts']:
            assert not (destination / route).exists(), route
        assert_no_mentions('ready-sentinel', 'draft-sentinel', 'future-sentinel',
                           'a-place-for-learning', 'sentinel-topic')
        assert not ET.parse(destination / 'feed.xml').findall('{http://www.w3.org/2005/Atom}entry')

    build()
    assert_off()
    off_home = read(destination / 'index.html')
    print('PASS: off omits blog pages, navigation, feed and sitemap entries.')

    build(enabled=True)
    assert (destination / 'blog/ready-sentinel/index.html').exists()
    assert not (destination / 'blog/draft-sentinel/index.html').exists()
    assert not (destination / 'blog/future-sentinel/index.html').exists()
    assert not (destination / 'blog/a-place-for-learning/index.html').exists()
    assert 'ready-sentinel' in read(destination / 'feed.xml')
    assert 'ready-sentinel' in read(destination / 'sitemap.xml')
    assert 'ready-sentinel' in read(destination / 'tags/index.html')
    assert 'Recent Posts' in read(destination / 'index.html')
    assert 'data-load-comments' not in read(destination / 'blog/ready-sentinel/index.html')
    assert_no_mentions('draft-sentinel', 'future-sentinel', 'a-place-for-learning', 'Blog Post number')
    print('PASS: on publishes ready entries only; drafts and future entries stay out.')

    build(preview=True, configured=True)
    assert (destination / 'blog/draft-sentinel/index.html').exists()
    assert (destination / 'blog/a-place-for-learning/index.html').exists()
    assert not (destination / 'blog/future-sentinel/index.html').exists()
    archive = read(destination / 'blog/index.html')
    assert archive.index('a-place-for-learning') < archive.index('draft-sentinel') < archive.index('ready-sentinel')
    article = read(destination / 'blog/a-place-for-learning/index.html')
    assert 'blog-toc' in article and 'assets/js/blog.js' in article
    assert 'Back to Blog' in article and 'language-python' in article
    assert 'data-mode="preview"' in article and 'data-server=""' in article
    assert 'data-load-comments' not in article
    assert 'comments.example.test' not in article, 'Preview leaked the live endpoint'
    assert_no_mentions('future-sentinel', 'Blog Post number')
    print('PASS: preview shows unfinished entries in date order with article formatting.')

    build(enabled=True, configured=True)
    ready = read(destination / 'blog/ready-sentinel/index.html')
    assert 'data-mode="live"' in ready and 'data-load-comments' in ready
    assert 'data-path="/blog/ready-sentinel/"' in ready
    assert 'blog-comment-form' not in ready
    assert 'blog-interactions.js' not in read(destination / 'index.html')
    print('PASS: live interactions load only on published articles with a configured endpoint.')

    post = source / '_blog/ready-sentinel.md'
    post.write_text(read(post).replace('published: true', 'published: true\ninteractions: false'))
    build(enabled=True, configured=True)
    assert 'blog-interactions.js' not in read(destination / 'blog/ready-sentinel/index.html')
    build(enabled=True, configured=True, interactions=False)
    assert_no_mentions('data-load-comments', 'data-mode="live"')
    print('PASS: per-post and global interaction switches disable the widget.')

    build()
    assert_off()
    # The existing academic content survives the on -> preview -> off sequence.
    assert read(destination / 'index.html').split('id="publications"', 1)[1] == off_home.split('id="publications"', 1)[1]
    print('PASS: switching off cleans previous blog output and preserves homepage sections.')
