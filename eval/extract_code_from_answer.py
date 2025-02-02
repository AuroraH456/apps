# extracts pure code from text answer
import re

def extract_content_between_keys(response, start_key, end_key):
      # Define a regular expression pattern to match content between start_key and end_key
      pattern = re.compile(re.escape(start_key) + r'(.*?)' + re.escape(end_key), re.DOTALL)

      # Find all matches of the pattern
      matches = pattern.findall(response)

      if not matches:
          print(f"No content found between '{start_key}' and '{end_key}'.")
          return

      # Join all matched content
      response = ''.join(matches)

      print(f"Content between '{start_key}' and '{end_key}' has been extracted.")
      return response

#response = 'abcdefghij'
start_key = '```python'
end_key = '```'
#response = extract_content_between_keys(response, start_key, end_key)
#print(response)
