import madgrad

lr = [.002, .001, .0005]

results = pd.DataFrame(columns = ['optimizer', 'train_acc', 'train_loss', 'test_acc', 'test_loss'])

for rate in lr:
  for optim_name in ['Adam', 'Madgrad']:
    model = get_cnn_model()
    if optim_name == 'Adam':
      optimizer = optim.Adam(model.parameters(), lr = rate)
    else:
      optimizer = madgrad.MADGRAD(model.parameters(), lr = rate)
    train_acc, train_loss = train_cnn(model, train_iterator, optimizer, criterion)
    test_acc, test_loss = test_cnn_model(model, test_iterator, criterion)
    print('\n---------------------------------------------')
    print('%s AND LR OF %.4f RESULTS' % (optim_name, rate))
    print('TRAIN ACC: %.4f, TRAIN LOSS: %.4f' % (train_acc, train_loss))
    print('TEST ACC: %.4f, TEST LOSS: %.4f' % (test_acc, test_loss))
    print('---------------------------------------------\n')

    results.append({'optimizer': optim_name,
                    'train_acc' : train_acc,
                    'train_loss': train_loss,
                    'test_acc': test_acc,
                    'test_loss': test_loss}, ignore_index = True)
print(results)
