import cv2
import numpy as np


def detectar_zona_luz(frame, limiar=25, margem=10):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv2.threshold(blurred, limiar, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contornos_validos = [c for c in contours if cv2.contourArea(c) > 100]
    if not contornos_validos:
        return None

    todos_pontos = np.vstack(contornos_validos)
    x, y, w, h = cv2.boundingRect(todos_pontos)

    h_img, w_img = frame.shape[:2]
    return (
        max(0, x - margem),
        max(0, y - margem),
        min(w_img, x + w + margem),
        min(h_img, y + h + margem),
    )


def desenhar_grafico_opencv(
    largura, altura, r_dist, g_dist, b_dist, total_dist, estado_botoes
):
    painel = np.ones((altura, largura, 3), dtype=np.uint8) * 255

    cv2.rectangle(painel, (0, 0), (largura, 60), (35, 35, 35), -1)

    botoes = [
        (
            'RETOMAR' if estado_botoes['pausado'] else 'PAUSAR',
            (80, 12, 270, 48),
            (40, 40, 198) if estado_botoes['pausado'] else (0, 108, 239),
        ),
        ('SALVAR PNG', (330, 12, 520, 48), (154, 27, 106)),
    ]

    for texto, (x1, y1, x2, y2), cor in botoes:
        cv2.rectangle(painel, (x1, y1), (x2, y2), cor, -1)
        cv2.rectangle(painel, (x1, y1), (x2, y2), (200, 200, 200), 1)

        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        tx = x1 + (x2 - x1 - tw) // 2
        ty = y1 + (y2 - y1 + th) // 2
        cv2.putText(
            painel,
            texto,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    margem_esq, margem_dir = 55, 25
    margem_sup, margem_inf = 90, 45
    g_w = largura - margem_esq - margem_dir
    g_h = altura - margem_sup - margem_inf

    cv2.rectangle(
        painel,
        (margem_esq, margem_sup),
        (margem_esq + g_w, margem_sup + g_h),
        (180, 180, 180),
        1,
    )

    for val in [0, 50, 100, 150, 200, 250]:
        y = int(margem_sup + g_h - (val / 260.0) * g_h)
        cv2.line(
            painel, (margem_esq, y), (margem_esq + g_w, y), (235, 235, 235), 1
        )
        cv2.putText(
            painel,
            str(val),
            (15, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (80, 80, 80),
            1,
        )

    cv2.putText(
        painel,
        'Perfil Espectral em Tempo Real',
        (margem_esq, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )

    if len(r_dist) > 1:
        n = len(r_dist)
        xs = np.linspace(margem_esq, margem_esq + g_w, n).astype(np.int32)

        def plotar_linha(dados, cor, espessura=2):
            ys = (
                margem_sup + g_h - (np.clip(dados, 0, 260) / 260.0) * g_h
            ).astype(np.int32)
            pts = np.column_stack((xs, ys)).reshape((-1, 1, 2))
            cv2.polylines(
                painel, [pts], isClosed=False, color=cor, thickness=espessura
            )

        plotar_linha(total_dist, (0, 200, 200), 2)
        plotar_linha(b_dist, (255, 0, 0), 2)
        plotar_linha(g_dist, (0, 180, 0), 2)
        plotar_linha(r_dist, (0, 0, 255), 2)

        lx = margem_esq + g_w - 115
        ly = margem_sup + 15
        itens_legenda = [
            ('Total', (0, 200, 200)),
            ('Azul', (255, 0, 0)),
            ('Verde', (0, 180, 0)),
            ('Vermelho', (0, 0, 255)),
        ]
        for idx, (nome, cor) in enumerate(itens_legenda):
            y_item = ly + idx * 18
            cv2.line(painel, (lx, y_item), (lx + 18, y_item), cor, 2)
            cv2.putText(
                painel,
                nome,
                (lx + 24, y_item + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (40, 40, 40),
                1,
            )

    return painel, botoes


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    estado = {'pausado': False, 'clique': None}
    display_cam = None
    painel_atual = None
    botoes_coords = []

    LARGURA_LADO = 600
    ALTURA_JANELA = 480

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x >= LARGURA_LADO:
                px = x - LARGURA_LADO
                for nome, (bx1, by1, bx2, by2), _ in botoes_coords:
                    if bx1 <= px <= bx2 and by1 <= y <= by2:
                        estado['clique'] = nome

    nome_janela = 'Espectrometro em Tempo Real'
    cv2.namedWindow(nome_janela)
    cv2.setMouseCallback(nome_janela, on_mouse)

    r_dist, g_dist, b_dist, total_dist = [], [], [], []

    while True:
        if estado['clique']:
            acao = estado['clique']
            estado['clique'] = None

            if acao in ('PAUSAR', 'RETOMAR'):
                estado['pausado'] = not estado['pausado']
            elif acao == 'SALVAR PNG' and painel_atual is not None:
                cv2.imwrite('espectro_grafico.png', painel_atual[60:, :])
                print("Gráfico salvo como 'espectro_grafico.png'")

        if not estado['pausado'] or display_cam is None:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_redim = cv2.resize(frame, (LARGURA_LADO, ALTURA_JANELA))
            display_cam = frame_redim.copy()

            bbox = detectar_zona_luz(frame_redim)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cropped = frame_redim[y1:y2, x1:x2]
                cv2.rectangle(display_cam, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if cropped.size > 0:
                    kernel = np.ones(5) / 5.0
                    b_dist = np.convolve(
                        np.mean(cropped[:, :, 0], axis=1), kernel, mode='same'
                    )
                    g_dist = np.convolve(
                        np.mean(cropped[:, :, 1], axis=1), kernel, mode='same'
                    )
                    r_dist = np.convolve(
                        np.mean(cropped[:, :, 2], axis=1), kernel, mode='same'
                    )
                    total_dist = np.clip(b_dist + g_dist + r_dist, 0, 255)
            else:
                r_dist, g_dist, b_dist, total_dist = [], [], [], []

        painel_atual, botoes_coords = desenhar_grafico_opencv(
            LARGURA_LADO,
            ALTURA_JANELA,
            r_dist,
            g_dist,
            b_dist,
            total_dist,
            estado,
        )

        tela_final = np.hstack((display_cam, painel_atual))
        cv2.imshow(nome_janela, tela_final)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
