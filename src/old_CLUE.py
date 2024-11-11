class CLUESamplingSegmentation:
#     """
#     Implements CLUE for Segmentation: Clustering via Uncertainty-weighted Embeddings at pixel-level.
#     """
#     def __init__(self, dset, model, device, args, cache_path):
#         self.dset = dset
#         self.model = model
#         self.device = device
#         self.args = args
#         self.T = args.clue_softmax_t
#         self.cache_path = cache_path

#     def get_pixel_embeddings(self, model, loader, device, emb_dim=512, buffer_size=50):
#         if os.path.exists(self.cache_path):
#             print(f"Loading embeddings and scores from cache: {self.cache_path}")
#             with open(self.cache_path, "rb") as f:
#                 embeddings, pixel_scores = pickle.load(f)
#             return embeddings, pixel_scores

#         print("Extracting pixel embeddings and uncertainty scores (this may take some time)...")
#         model.eval()

#         temp_dir = "temp_files"
#         os.makedirs(temp_dir, exist_ok=True)

#         embeddings_buffer = []
#         scores_buffer = []
#         file_index = 0

#         def save_buffer_to_temp_files():
#             """Сохраняет буферные данные в отдельные временные файлы."""
#             nonlocal file_index
#             if embeddings_buffer:
#                 np.save(f"{temp_dir}/embeddings_{file_index}.npy", np.concatenate(embeddings_buffer))
#                 np.save(f"{temp_dir}/scores_{file_index}.npy", np.concatenate(scores_buffer))
#                 embeddings_buffer.clear()
#                 scores_buffer.clear()
#                 file_index += 1

#         with torch.no_grad():
#             for i, batch in enumerate(tqdm(loader, desc="Extracting Pixel Embeddings")):
#                 data = batch["data"].to(device)

#                 # Извлекаем эмбеддинги

#                 # pred: (B, C, H, W) = (32, 4, 256, 256)
#                 # emb_pen: (B, C, H, W) = (32, 32, 128, 128)
#                 pred, emb_pen = model(data, with_emb=True)
                
#                 emb_pen = emb_pen.permute(0, 2, 3, 1).reshape(-1, emb_pen.shape[1])
#                 pred = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1]).cpu()
                
#                 # Вычисляем неопределённость
#                 probs = nn.Softmax(dim=1)(pred / self.T)
#                 probs += 1e-8
#                 uncertainty = -(probs * torch.log(probs)).sum(dim=1).cpu().numpy().flatten()

#                 if emb_pen.shape[0] != uncertainty.shape[0]:
#                     print("Filtering embeddings based on uncertainty...")
                    
#                     # Сортируем индексы по возрастанию неопределённости
#                     sorted_indices = np.argsort(uncertainty)
                    
#                     # Определяем, сколько элементов нужно удалить
#                     num_to_remove = abs(emb_pen.shape[0] - uncertainty.shape[0])
                    
#                     filtered_indices = sorted_indices[num_to_remove:]
#                     if emb_pen.shape[0] > uncertainty.shape[0]:
#                         # Удаляем наименее информативные эмбеддинги
#                         embeddings = embeddings[filtered_indices]
#                     else:
#                         # Удаляем элементы неопределённости с наименьшими значениями
#                         uncertainty = uncertainty[filtered_indices]

#                 # Проверка соответствия размеров
#                 assert emb_pen.shape[0] == uncertainty.shape[0], \
#                     f"Size mismatch: embeddings ({emb_pen.shape[0]}) vs uncertainty ({uncertainty.shape[0]})"
#                 # Добавляем данные в буферы
#                 embeddings_buffer.append(emb_pen.cpu().numpy())
#                 scores_buffer.append(uncertainty)

#                 # Записываем данные в файл каждые `buffer_size` итераций
#                 if (i + 1) % buffer_size == 0:
#                     save_buffer_to_temp_files()
#                     break

#         # Сохранить остаток буфера
#         save_buffer_to_temp_files()

#         # Объединяем все временные файлы в один
#         embeddings = []
#         pixel_scores = []

#         print("Merging temporary files...")
#         embeddings_files = sorted(glob.glob(f"{temp_dir}/embeddings_*.npy"))
#         scores_files = sorted(glob.glob(f"{temp_dir}/scores_*.npy"))

#         for emb_file, score_file in tqdm(zip(embeddings_files, scores_files), total=len(embeddings_files)):
#             embeddings.append(np.load(emb_file))
#             pixel_scores.append(np.load(score_file))

#         embeddings = np.concatenate(embeddings, axis=0)
#         pixel_scores = np.concatenate(pixel_scores, axis=0)

#         # Сохраняем объединённые данные в кеш
#         with open(self.cache_path, "wb") as f:
#             pickle.dump((embeddings, pixel_scores), f)
#         print(f"Saved embeddings and scores to cache: {self.cache_path}")

#         # # Удаляем временные файлы
#         # for file in embeddings_files + scores_files:
#         #     os.remove(file)
#         # os.rmdir(temp_dir)

#         return embeddings, pixel_scores

#     def query(self, n_samples, loader):
#         # Извлекаем эмбеддинги и неопределённость пикселей
#         pixel_embeddings, pixel_scores = self.get_pixel_embeddings(self.model, loader, self.device)

#         # Применяем K-Means, взвешивая по неопределённости
#         kmeans = KMeans(n_clusters=n_samples)
#         kmeans.fit(pixel_embeddings, sample_weight=pixel_scores)

#         dists = euclidean_distances(kmeans.cluster_centers_, pixel_embeddings)
#         sort_idxs = dists.argsort(axis=1)
#         selected_indices = []
#         ax, rem = 0, n_samples
#         while rem > 0:
#             selected_indices.extend(list(sort_idxs[:, ax][:rem]))
#             selected_indices = list(set(selected_indices))  # Убираем повторения
#             rem = n_samples - len(selected_indices)
#             ax += 1
#         return selected_indices
