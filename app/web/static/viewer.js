window.renderWithBoxes = async function(imgSrc, boxes){
    const img = document.getElementById('baseimg');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
  
    img.onload = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.drawImage(img, 0, 0);
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(239,68,68,0.95)'; // rojo
      boxes.forEach(b=>{
        ctx.strokeRect(b.x, b.y, b.w, b.h);
      });
    };
    img.src = imgSrc;
  }
  