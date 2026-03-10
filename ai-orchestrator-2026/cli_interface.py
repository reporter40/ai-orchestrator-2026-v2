"""
cli_interface.py — Интерактивный CLI для AI Orchestrator 2026
============================================================
Обеспечивает удобный ввод параметров через терминал.
"""

from __future__ import annotations

import os
from typing import Optional


def print_banner():
    print("\n" + "=" * 70)
    print("🚀 AI ORCHESTRATOR 2026 — INTERACTIVE CONSOLE")
    print("=" * 70)


async def run_cli() -> tuple[str, Optional[str]]:
    """
    Запустить интерактивный опрос пользователя.
    
    Returns:
        (request, external_url)
    """
    print_banner()
    
    print("\n[1] Введите ваш запрос для генерации промпта:")
    request = input("> ").strip()
    
    if not request:
        request = "Создай креативный промпт для космической оперы."
        print(f"Используем запрос по умолчанию: {request}")

    print("\n[2] Выберите режим работы:")
    print("    1. Автоматический (MoE Pool)")
    print("    2. Внешний агент (по URL)")
    mode = input("Выбор (1/2): ").strip()
    
    external_url = None
    if mode == "2":
        print("\n[3] Введите URL внешнего агента (например, GenSpark):")
        external_url = input("> ").strip()
        if not external_url:
            external_url = "http://localhost:8080/generate"
            print(f"Используем URL по умолчанию: {external_url}")

    print("\n" + "-" * 70)
    print("⚙️ Запуск оркестрации...")
    print("-" * 70 + "\n")
    
    return request, external_url
