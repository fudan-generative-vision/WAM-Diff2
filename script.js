const header = document.querySelector('.site-header');
const menuButton = document.querySelector('.menu-button');
const mobileNav = document.querySelector('.mobile-nav');
const toast = document.querySelector('.toast');
const copyButton = document.querySelector('.copy-button');
const lightbox = document.querySelector('.lightbox');
const lightboxImage = lightbox.querySelector('img');
let toastTimer;

function updateHeader() {
  header.classList.toggle('scrolled', window.scrollY > 20);
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add('show');
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toast.classList.remove('show'), 2200);
}

updateHeader();
window.addEventListener('scroll', updateHeader, { passive: true });

menuButton.addEventListener('click', () => {
  const open = menuButton.getAttribute('aria-expanded') === 'true';
  menuButton.setAttribute('aria-expanded', String(!open));
  mobileNav.classList.toggle('open', !open);
});

mobileNav.querySelectorAll('a').forEach((link) => {
  link.addEventListener('click', () => {
    menuButton.setAttribute('aria-expanded', 'false');
    mobileNav.classList.remove('open');
  });
});

document.querySelectorAll('[data-placeholder]').forEach((link) => {
  link.addEventListener('click', (event) => {
    event.preventDefault();
    showToast(`${link.dataset.placeholder} coming soon.`);
  });
});

copyButton.addEventListener('click', async () => {
  const bibtex = document.querySelector('#bibtex').textContent;
  try {
    await navigator.clipboard.writeText(bibtex);
    copyButton.querySelector('span').textContent = 'Copied';
    showToast('BibTeX copied to clipboard.');
    window.setTimeout(() => { copyButton.querySelector('span').textContent = 'Copy'; }, 1800);
  } catch {
    showToast('Select the citation text to copy it.');
  }
});

document.querySelectorAll('[data-lightbox]').forEach((button) => {
  button.addEventListener('click', () => {
    lightboxImage.src = button.dataset.lightbox;
    lightboxImage.alt = button.querySelector('img').alt;
    lightbox.showModal();
    document.body.classList.add('no-scroll');
  });
});

function closeLightbox() {
  lightbox.close();
  document.body.classList.remove('no-scroll');
  lightboxImage.src = '';
}

document.querySelector('.lightbox-close').addEventListener('click', closeLightbox);
lightbox.addEventListener('click', (event) => {
  if (event.target === lightbox) closeLightbox();
});
lightbox.addEventListener('close', () => document.body.classList.remove('no-scroll'));

const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.08, rootMargin: '0px 0px -35px' });

document.querySelectorAll('.reveal').forEach((element) => observer.observe(element));
