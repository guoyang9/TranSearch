import numpy as np

import torch
import torch.nn.functional as F


def mrr(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(float(index+1))
	else:
		return 0


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, data_test, dataloader_test, top_k, is_output, epoch):
	Mrr, Hr, Ndcg = [], [], []

	######################## COMPUTE ALL ITEMS ONCE ####################
	all_items_embed = []
	all_item_idxs = []
	for vis, text, asin in data_test.get_all_test():
		vis = torch.cuda.FloatTensor(vis).view(1, -1)
		text = torch.cuda.FloatTensor(text).view(1, -1)
		item_embed = model(None, None, vis, text, None, None, True)

		all_item_idxs.append(asin)
		all_items_embed.append(item_embed.view(-1).data.cpu().numpy())

	item_size = len(all_item_idxs)
	all_items_embed = torch.cuda.FloatTensor(np.array(all_items_embed))

	all_items_map = {i: item for i, item in enumerate(all_item_idxs)}
	user_bought = data_test.user_bought

	####################### FOR EVERY USER-QUERY PAIR ####################
	for idx, batch_data in enumerate(dataloader_test):
		user = batch_data['userID'].cuda()
		query = batch_data['query'].cuda()
		reviewerID = batch_data['reviewerID'][0]
		item = batch_data['item'][0]
		query_text = batch_data['query_text']

		item_predict = model(user, query, None, None, None, None, False)

		scores = F.pairwise_distance(
					item_predict.repeat(item_size, 1), all_items_embed)

		_, ranking_list = scores.sort(dim=-1, descending=True)
		ranking_list = [all_items_map[i] for i in ranking_list.tolist()]
		top_idx = []
		u_bought = user_bought[reviewerID]
		while len(top_idx) < top_k:
			candidate_item = ranking_list.pop()
			if candidate_item not in u_bought or candidate_item == item:
				top_idx.append(candidate_item)

		Mrr.append(mrr(item, top_idx))
		Hr.append(hit(item, top_idx))
		Ndcg.append(ndcg(item, top_idx))

	return np.mean(Mrr), np.mean(Hr), np.mean(Ndcg)
