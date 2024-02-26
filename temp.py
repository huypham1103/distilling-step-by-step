from linkedin_api import Linkedin

# Authenticate using any Linkedin account credentials
api = Linkedin('huyquochoan7@gmail.com', '123456789huy')

# GET a profile
profile = api.get_profile('Dr. Christof Henkel')
# profile = api.get_profile('abbyspeicher')
# post = api.get_profile_posts('abbyspeicher')
# detail = api.get_post_comments(post_urn='urn:li:activity:6505933340296777728')

# post = api.search({'keywords': 'hiring', 'template': 'CONTENT_A'}, limit=100)
# GET a profiles contact info
# contact_info = api.get_profile_contact_info('abbyspeicher')
# api.get_feed_posts
# temp = api.search_people(connection_of='ACoAAA8-pd0BA8H2los1nsKbUfIFJ6L10MPYqgc')
# GET 1st degree connections of a given profile
# connections = api.get_profile_connections('1234asc12304')
# temp = api.search_companies(keywords='Delhi University')
# temp = api.search_people(keyword_company='Trevena, Inc.')
# temp = api.search_people(schools='79825902')
print(profile)
print(api.get_profile_contact_info('dr-christof-henkel-766a54ba'))
