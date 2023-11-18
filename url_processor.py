def is_arxiv_url(url: str) -> bool:
    import requests
    import re
    import urllib
    from typing import Optional, Union
    arxiv_pattern = r'https?://arxiv\.org/abs/.+'
    return bool(re.match(arxiv_pattern, url))


def is_acl_anthology_url(url: str) -> bool:
    import requests
    import re
    import urllib
    from typing import Optional, Union
    acl_anthology_pattern = r'https://aclanthology\.org/.*?/'
    return bool(re.match(acl_anthology_pattern, url))


def url_processor(url: str) -> str:
    import requests
    import re
    import urllib
    from typing import Optional, Union

    class DownloadError(Exception):
        pass

    class InvalidURLException(Exception):
        pass
    paper_url = url

    if is_arxiv_url(paper_url):
        pdf_url = paper_url.replace('/abs/', '/pdf/') + '.pdf'
    elif is_acl_anthology_url(paper_url):
        pdf_url = paper_url.rstrip('/') + '.pdf'
    else:
        raise InvalidURLException('Invalid URL. Please provide a valid ArXiv or ACL Anthology URL.')
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()  # Raise an exception if there's an HTTP error

        pdf_filename = pdf_url.split('/')[-1]

        with open(pdf_filename, 'wb') as pdf_file:
            pdf_file.write(response.content)

    except requests.exceptions.RequestException as e:
        raise DownloadError(f'Failed to download the PDF: {e}')

    filename = pdf_filename
    return filename

