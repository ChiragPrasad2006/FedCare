import sys

with open('hospital_server/templates/dashboard.html', 'r', encoding='utf-8') as f:
    text = f.read()

start_marker = '                <section class="panel" style="margin-top: 18px;">\n                <div class="muted">Training</div>'
end_marker = '                </div>\n    </section>\n'

start_idx = text.find(start_marker)
if start_idx == -1:
    print('Start not found')
    sys.exit(1)

end_idx = text.find('    </section>\n', start_idx) + len('    </section>\n')

training_block = text[start_idx:end_idx]
text = text[:start_idx] + text[end_idx:]

insert_marker = '        <section class="grid">'
insert_idx = text.find(insert_marker)

# We should fix the indentation of training_block so it looks nice
# Currently it starts with 16 spaces for <section, and then 16 for <div>
# Let's just wrap it nicely
training_block = training_block.replace('                <section', '        <section').replace('    </section>', '        </section>')

text = text[:insert_idx] + training_block + '\n' + text[insert_idx:]

with open('hospital_server/templates/dashboard.html', 'w', encoding='utf-8') as f:
    f.write(text)

print('Done!')
