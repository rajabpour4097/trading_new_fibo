FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99
ENV WINEPREFIX=/home/wineuser/.wine
ENV WINEARCH=win64
ENV TZ=Asia/Tehran

# Sys deps
RUN apt-get update && apt-get install -y \
    locales tzdata ca-certificates wget curl unzip \
    xvfb x11vnc fluxbox supervisor \
    software-properties-common gnupg2 cabextract p7zip-full \
    python3 python3-pip python3-venv \
    wine64 winetricks \
  && rm -rf /var/lib/apt/lists/*

# Locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# User for Wine
RUN useradd -m -s /bin/bash wineuser
USER wineuser
WORKDIR /home/wineuser

# Initialize Wine and deps for MT5
RUN wine wineboot --init || true
RUN winetricks -q corefonts vcrun2019

# Back to root to copy app
USER root
WORKDIR /app
COPY . /app

# Python deps (از requirements سطح پروژه استفاده می‌کنیم)
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# دانلود نصب‌کننده MT5 (اجرای آن در اولین بار توسط VNC انجام می‌شود)
RUN wget -O /app/mt5setup.exe https://download.mql5.com/cdn/web/metaquotes.ltd/mt5/mt5setup.exe

# Supervisor config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# دسترسی‌ها
RUN chown -R wineuser:wineuser /app
RUN mkdir -p /var/log/supervisor && chown -R wineuser:wineuser /var/log/supervisor

EXPOSE 5900

# اجرا
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]