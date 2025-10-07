// ====== static/main.js ======
'use strict';

const $ = (s, p = document) => p.querySelector(s);

// >>>>>> ตั้งค่า URL ของ API (แก้ให้ตรงสภาพแวดล้อมคุณ) <<<<<<
// ถ้าหน้าเว็บกับ API อยู่โดเมนเดียวกัน:  const API_BASE = '';
// ถ้า API อยู่คนละโดเมน/พอร์ต: ใส่เต็ม เช่น:
// const API_BASE = 'http://localhost:4400';           // ตอนทดสอบในเครื่อง
const API_BASE = 'https://g5weds.consolutechcloud.com/backend/';   // ตอนขึ้นเซิร์ฟเวอร์จริง

const PREDICT_URL = `${API_BASE}/predict`;
const HEALTH_URL = `${API_BASE}/health`;

const ACCEPT = ['image/png','image/jpeg'];
const MAX_SIZE = 10 * 1024 * 1024; // 10MB

function setBusy(el, busy=true){ el?.setAttribute('aria-busy', String(busy)); }

function showStatus(msg, kind='info'){
  const el = $('#status');
  el.innerHTML = kind === 'loading' ? `<span class="loading"><span class="spinner"></span>${msg}</span>` : msg;
}

function setResultText(text){
  const box = $('#result');
  const out = $('#resultText');
  out.textContent = text || '';
  box.hidden = !text;
}

function validateFile(file){
  if(!file) return { ok:false, reason:'โปรดเลือกไฟล์ภาพ' };
  if(!ACCEPT.includes(file.type)) return { ok:false, reason:'รองรับเฉพาะไฟล์ PNG หรือ JPG' };
  if(file.size > MAX_SIZE) return { ok:false, reason:'ไฟล์ใหญ่เกิน 10MB' };
  return { ok:true };
}

async function processInput(){
  const fileInput = $('#file');
  const file = fileInput.files?.[0] || null;

  // ขณะนี้ API ต้องการภาพเป็นหลัก
  const v = validateFile(file);
  if(!v.ok){ showStatus(v.reason); return; }

  showStatus('กำลังประมวลผล...', 'loading');
  setBusy($('#result'), true);
  setResultText('');
  $('#resultImage').removeAttribute('src');

  try{
    // เรียก /predict ฝั่ง Flask ด้วย multipart/form-data
    const fd = new FormData();
    fd.append('image', file);
    // ถ้าอยากส่งข้อความประกอบไปด้วย สามารถ fd.append('note', text) แล้วปรับฝั่ง API ตามต้องการ

    // ✅ ใช้ URL แบบ absolute ไปยัง Flask จริง
    const res = await fetch(PREDICT_URL, {
      method: 'POST',
      body: fd,
      // ถ้า API เปิด CORS แล้วพอ ไม่ต้องตั้ง header เพิ่ม
      // mode: 'cors',
      // credentials: 'omit',
    });
    if(!res.ok){
      // พยายามอ่านข้อความ error จาก API เพื่อ debug ง่ายขึ้น
      let errText = res.statusText;
      try { const j = await res.json(); errText = j.error || errText; } catch {}
      throw new Error(`${res.status} ${errText}`);
    }
    const data = await res.json();

    // สร้างข้อความสรุปสวย ๆ
    const gens = (data.inferred?.generics || []).join(', ') || '-';
    const strs = (data.inferred?.strengths || []).join(', ') || '-';
    const matches = (data.matched_products || []).map(m => m.match_type).join(', ') || '-';

    const detailText = [
      `Generics: ${gens}`,
      `Strengths: ${strs}`,
      `Match type: ${matches}`
    ].join('\n');

    setResultText(detailText);

    // แสดงภาพ (มีกรอบ +สรุปด้านล่าง)
    if(data.image_base64){
      $('#resultImage').src = data.image_base64;
    }

    showStatus('เสร็จสิ้น ✔');
  }catch(err){
    console.error(err);
    showStatus('เกิดข้อผิดพลาด: ' + String(err.message||err));
    setResultText(`เกิดข้อผิดพลาด: ${String(err?.message||err)}`);
  }finally{
    setBusy($('#result'), false);
  }
}

function setupDrop(){
  const dz = $('#drop');
  const fi = $('#file');
  const hint = $('#fileHint');

  ['dragenter','dragover'].forEach(ev=>{
    dz.addEventListener(ev, e=>{ e.preventDefault(); dz.classList.add('drag'); });
  });
  ;['dragleave','drop'].forEach(ev=>{
    dz.addEventListener(ev, e=>{ e.preventDefault(); dz.classList.remove('drag'); });
  });
  dz.addEventListener('drop', e => {
    const f = e.dataTransfer.files?.[0];
    if(f){
      const v = validateFile(f);
      if(!v.ok){ showStatus(v.reason); return; }
      const dt = new DataTransfer(); dt.items.add(f); fi.files = dt.files;
      hint.textContent = `เลือกไฟล์: ${f.name}`;
    }
  });
  fi.addEventListener('change', ()=>{
    const f = fi.files?.[0];
    hint.textContent = f ? `เลือกไฟล์: ${f.name}` : 'ยังไม่ได้เลือกไฟล์';
  });
  $('#clearFile').addEventListener('click', ()=>{ fi.value=''; hint.textContent='ยังไม่ได้เลือกไฟล์'; });
}

function setup(){
  $('#btnProcess').addEventListener('click', processInput);
  $('#btnReset').addEventListener('click', ()=>{
    $('#question').value = '';
    $('#file').value = '';
    $('#fileHint').textContent = 'ยังไม่ได้เลือกไฟล์';
    setResultText('');
    showStatus('');
    $('#resultImage').removeAttribute('src');
  });
  $('#clearText').addEventListener('click', ()=>{ $('#question').value = ''; });
  $('#copyText').addEventListener('click', async ()=>{
    try{
      await navigator.clipboard.writeText($('#resultText').textContent || '');
      showStatus('คัดลอกผลลัพธ์แล้ว');
    }catch{ showStatus('คัดลอกไม่สำเร็จ'); }
  });

  setupDrop();
}

document.addEventListener('DOMContentLoaded', setup);
