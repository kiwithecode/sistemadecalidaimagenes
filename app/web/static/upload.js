const drop = document.getElementById('drop');
const filepick = document.getElementById('filepick');
const list = document.getElementById('list');
const sendBtn = document.getElementById('send');
const skuInput = document.getElementById('sku');
const log = document.getElementById('log');

let files = [];

function renderList(){
  list.innerHTML = '';
  files.forEach(f=>{
    const li = document.createElement('li');
    li.textContent = `${f.name} (${Math.round(f.size/1024)} KB)`;
    list.appendChild(li);
  });
}

drop.addEventListener('click', ()=> filepick.click());
drop.addEventListener('dragover', e=>{ e.preventDefault(); drop.style.opacity=0.8; });
drop.addEventListener('dragleave', e=>{ drop.style.opacity=1; });
drop.addEventListener('drop', e=>{
  e.preventDefault(); drop.style.opacity=1;
  files = [...files, ...e.dataTransfer.files];
  renderList();
});
filepick.addEventListener('change', e=>{
  files = [...files, ...e.target.files];
  renderList();
});

sendBtn.addEventListener('click', async ()=>{
  if(!files.length){ log.textContent = 'Selecciona al menos una imagen.'; return; }
  const fd = new FormData();
  files.forEach(f=> fd.append('files', f));
  const sku = (skuInput.value||'auto').trim();
  log.textContent = 'Subiendo...';
  try{
    const r = await fetch(`/batch/predict?sku=${encodeURIComponent(sku)}`, { method:'POST', body: fd });
    const data = await r.json();
    if(!r.ok){ log.textContent = `Error: ${data.error||r.status}`; return; }
    // redirigir al lote en el dashboard
    window.location.href = `/web/batch/${data.lote}`;
  }catch(err){
    log.textContent = 'Fallo la subida: ' + err.message;
  }
});
