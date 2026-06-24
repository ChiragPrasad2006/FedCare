import sys

with open('hospital_server/templates/dashboard.html', 'r', encoding='utf-8') as f:
    text = f.read()

start_marker = '            <section class="grid" style="grid-template-columns: 1fr; margin-top: 18px;">\n                <div class="panel">\n                    <div class="muted">Training</div>'
end_marker = '                    </div>\n                </div>\n            </section>\n'

start_idx = text.find(start_marker)
if start_idx == -1:
    print('Start not found')
    sys.exit(1)

end_idx = text.find('            </section>', start_idx) + len('            </section>\n')

training_block = text[start_idx:end_idx]
text = text[:start_idx] + text[end_idx:]

insert_marker = '        <section class="panel" style="margin-top: 18px;">\n            <div class="muted">Browse</div>'
insert_idx = text.find(insert_marker)

training_block = training_block.replace('<section class="grid" style="grid-template-columns: 1fr; margin-top: 18px;">\n                <div class="panel">', '        <section class="panel" style="margin-top: 18px;">')
training_block = training_block.replace('                </div>\n            </section>', '        </section>')

lines = training_block.split('\n')
new_lines = []
for line in lines:
    if line.startswith('    '):
        new_lines.append(line[4:])
    else:
        new_lines.append(line)
        
training_block = '\n'.join(new_lines)

text = text[:insert_idx] + training_block + '\n' + text[insert_idx:]

with open('hospital_server/templates/dashboard.html', 'w', encoding='utf-8') as f:
    f.write(text)

print('Done!')
