mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"le.minh.chien@sun-asterisk.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml